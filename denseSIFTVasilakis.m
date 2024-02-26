function features = denseSIFTVasilakis(inputds,Options)

arguments (Input)

    inputds                             {mustBeUnderlyingType(inputds, ...
                                                ['matlab.io.datastore.' ...
                                                'ImageDatastore'])}


    Options.Angles (1,1) double              {mustBeInRange(Options. ...
                                              Angles,1,36),...
                                              mustBeInteger,...
                                              mustBeNonnegative} = 8
    Options.numBins (1,1) double             {mustBeInRange(Options. ...
                                              numBins,1,36),...
                                              mustBeInteger,...
                                              mustBeNonnegative} = 4

    Options.Angle_Attenuation (1,1) double   {mustBeInteger,...
                                              mustBePositive} = 9

    Options.Grid_Spacing (1,1) double        {mustBeInteger,...
                                              mustBePositive} = 4

    Options.Patch_Size   (1,1) double        {mustBeInteger,...
                                              mustBePositive} = 16

    Options.Sigma_Edge   (1,:) double        {mustBePositive} = 1

end

fprintf("Using: \n Angles = %d \n Number of bins = %d \n Angle " + ...
        "attenuation = %d \n Grid Spacing = %d \n Patch size = %d \n" + ...
        " Sigma edge = %d\n\n",Options.Angles, ...
        Options.numBins,Options.Angle_Attenuation,Options.Grid_Spacing, ...
        Options.Patch_Size,Options.Sigma_Edge);

i=0;

for j = 1:length(inputds)
    img = read(inputds);
    [rows(j),cols(j)] = size(img);
end

min_row = min(rows,[],"all");
min_cols = min(cols,[],"all");

imgSizeThresh = [min_row,min_cols];

reset(inputds)
while hasdata(inputds)
    i=i+1;

    if mod(i,100)==0
        fprintf("Iteration: %d\n\n",i);
    end

    img = read(inputds);

    [hgt wid] = size(img);

    if hgt > imgSizeThresh(1) || wid > imgSizeThresh(2)
        %fprintf(['Image that is bigger than size threshold detected. ' ...
        %    'Processing:...\n']);
        img = imresize(img, imgSizeThresh, 'bicubic');
        %fprintf('Process ended\n');

    end

    [hgt wid] = size(img);

    %img = imresize(img,[200 400], 'bicubic');

    
    %% Initialization of the dense SIFT process!
    img = double(img);
    img = mean(img,3);
    img = img /max(img(:));
    
    num_samples = Options.numBins * Options.numBins;

    % Initializing the histogramic angles
    angle_step = 2 * pi / Options.Angles;
    angles = 0:angle_step:2*pi;
    angles(Options.Angles+1) = []; % bin centers

    % Generate dense gauss (review it)

    [GX,GY] = gaussVasilakis(Options.Sigma_Edge);

    % Add boundary
    img = [img(2:-1:1,:,:); img; img(end:-1:end-1,:,:)];
    img = [img(:,2:-1:1,:) img img(:,end:-1:end-1,:)];

    % Subtracting the mean intensity value from the original image. This is
    % a method of normalizing the image, meaning it enforces the mean
    % intensity value of the image to be zero.
    img = img-mean(img(:));

    % Apply convolution between the x and y gradient components with the
    % image, preserving the original image dimension
    I_X = filter2(GX, img, 'same'); % vertical edges
    I_Y = filter2(GY, img, 'same'); % horizontal edges

    I_X = I_X(3:end-2,3:end-2,:);
    I_Y = I_Y(3:end-2,3:end-2,:);

    % Calculation of the image-gradient magnitude through the vertical and
    % horizontal edges

    I_mag = sqrt(I_X.^2 + I_Y.^2); % gradient magnitude

    %% Calculation of the image-gradient through vertical and horizontal edges
    I_theta = atan2(I_Y,I_X);
    I_theta(find(isnan(I_theta))) = 0; % necessary????

    %% Forming the grid of the overall image.

    % grid
    grid_x = Options.Patch_Size/2:Options.Grid_Spacing:wid-...
        Options.Patch_Size/2+1;
    grid_y = Options.Patch_Size/2:Options.Grid_Spacing:hgt-...
        Options.Patch_Size/2+1;

    %% Initialize the size of the orientation image

    % make orientation images
    I_orientation = zeros([hgt, wid, Options.Angles], 'single');


    %% Calculation of the trigonometric numbers of the image-gradient

    % for each histogram angle
    cosI = cos(I_theta);
    sinI = sin(I_theta);

    %% Aligning the histogram angle with the angle of the image gradient and weighting with the image gradient magnitude
    for a=1:Options.Angles

        % compute each orientation channel

        % Calculation of the inner product between the respective
        %histogram angle (of the eight we are searching for) and the
        % orientation of the gradient.
        tmp = (cosI*cos(angles(a))+sinI*sin(angles(a))).^...
                                                Options.Angle_Attenuation;
        tmp = tmp .* (tmp > 0);

        % weight by magnitude
        I_orientation(:,:,a) = tmp .* I_mag;

        % For visualization and better perception of things: figure
        % imshow(I_orientation(:,:,a))


    end

    %% Apply spatial weighting to each orientation channel of the image.

    % Convolution formulation:

    % Initialize the weighting kernel with a size equal to the size of the
    % patch (here 16-by-16)
    weight_kernel = zeros(Options.Patch_Size,Options.Patch_Size);

    % Calculation of the patch centre.

    r = Options.Patch_Size/2; % Here the radius of the patch is
    % calculated, which is half the size of the patch.

    cx = r - 0.5;   % Here the x coordinate of the center of the
    % patch is calculated. The -0.5 part is used as the offset of the
    % center to align with the pixel grid. This is done because in many
    % image processing applications, it is assumed that the pixel grid
    % starts at the point (0.5,0.5).

    % Calculation of the resolution of sampling
    sample_res = Options.Patch_Size/Options.numBins;

    % A weighting vector is generated that represents the distance of each
    % pixel from the center of the patch, normalized by the sampling
    % resolution.
    weight_x = abs((1:Options.Patch_Size) - cx)/sample_res;

    % Apply a linear ramp-type function to the weighting vector so that it
    % decreases linearly from 1, from the center of the patch, to 0 at the
    % edges of the patch. Pixels beyond the boundaries of the patch have
    % zero weighting.
    weight_x = (1 - weight_x) .* (weight_x <= 1);

    for a = 1:Options.Angles

        % I_orientation(:,:,a) = conv2(I_orientation(:,:,a), weight_kernel,
        % 'same');

        % Convolve first by column of each angle channel orientation, with
        % weight_x and then convolve by row with weight_x' in order.
        I_orientation(:,:,a) = conv2(weight_x, weight_x', ...
            I_orientation(:,:,a),'same');
        % figure imshow(I_orientation(:,:,a))
    end

    %% SIFT sampling on valid points (no objects in the boundary)

    % Sample SIFT bins at valid locations (without boundary artifacts) find
    % coordinates of sample points (bin centers)

    % The meshgrid is a convenient and quick way to create a mesh for a
    % grid. Practically what it does is reproduce the values we have in the
    % 2 dimensions. Here, via linspace, the numbers are generated
    %
    %                       1     5     9    13    17
    %
    % So here:
    %
    % sample_x = | 1 	5	9	13	17 |
    %            | 1	5	9	13	17 | 
    %            | 1	5	9	13	17 | 
    %            | 1	5	9	13	17 | 
    %            | 1	5	9   13	17 |
    %
    % Conversely for sample_y:
    %
    % sample_y = | 1 	1	1	1	1  |
    %            | 5	5	5	5	5  | 
    %            | 9	9	9	9	9  | 
    %            | 13	13	13	13	13 | 
    %            | 17	17  17	17	17 |

    [sample_x, sample_y] = meshgrid(linspace(1,Options.Patch_Size+1,Options.numBins+1));
    %                                          ^^^^^^^^^^^^ ^^^^^^^^^^
    % Γιατί patch_size+1 και num_bins+1; Γιατί με τον τρόπο αυτό
    % εξασφαλίζουμε ότι το πλέγμα που δουλεύουμε θα έχει πάντοτε την σωστή
    % μορφή, είτε αυτή είναι άρτια (π.χ 4-by-4) είτε περιττή (π.χ 5-by-5).

    sample_x = sample_x(1:Options.numBins,1:Options.numBins);
    sample_x = sample_x(:)-Options.Patch_Size/2; % We convert the
    % sample_x matrix to a vector and then subtract by half

    % Why patch_size/2?
    %
    % This is how we shift the coordinates in the sample_x variable with
    % respect to the patch center.In the context of the code, the sample_x
    % table contains the coordinates on the x-axis of the sample points,
    % which are the centers of the SIFT bins. These coordinates are
    % initially set at the top-left corner of the patch with values ranging
    % in the range 1 to patch_size. So by subtracting patch_size/2 from
    % each coordinate, the coordinates are redefined relative to the center
    % of the patch. The center of the patch is now 0, while the range of
    % coordinates is changed to [-patch_size/2,patch_size/2]

    sample_y = sample_y(1:Options.numBins,1:Options.numBins);
    sample_y = sample_y(:)-Options.Patch_Size/2;

    %% Create a video showing dense image sampling
    % figure; imshow(I); hold on;
    %
    % Creating the image grid [grid_x, grid_y] =
    % meshgrid(1:Options.Grid_Spacing:size(I, 2), ...
    %    1:Options.Grid_Spacing:size(I, 1));
    %
    % Creation of the sampling grid [sample_x, sample_y] =
    % meshgrid(linspace(1, Options.Patch_Size, Options.numBins), ...
    %    linspace(1, Options.Patch_Size, Options.numBins));
    %
    % sample_x = sample_x(:) - Options.Patch_Size/2; sample_y = sample_y(:)
    % - Options.Patch_Size/2;
    %
    % Create a new display showing the zoomed version of the sampling grid
    %
    % figure;
    %
    % Create a VideoWriter object v =
    % VideoWriter('movingSamplingGrid.avi'); open(v);
    %
    % Plot the points of the sampling grid at each point of the image grid
    %
    %for i = 1:numel(grid_x)
    %    figure(1); clf; imshow(I); hold on; plot(grid_x, grid_y, 'r.'); %
    %    Visualization of the points of
    %                                  the image grid
    %
    %    Visualisation of the sampling grid at the underlying grid point
    %    plot(grid_x(i) + sample_x, grid_y(i) + sample_y, 'b.');
    %
    %   %% This part zooms in and shows how the sampling grid moves within
    %   the image grid
    %
    %   figure(2); clf; imshow(I); hold on; plot(grid_x(i) + sample_x,
    %   grid_y(i) + sample_y, 'b.');
    %
    %   Zoom in on the specific point of the image grid. axis([grid_x(i) -
    %   Options.Patch_Size, grid_x(i) + Options.Patch_Size, ...
    %     grid_y(i) - Options.Patch_Size, grid_y(i) + Options.Patch_Size]);
    %
    %
    %    frame = getframe(gcf); writeVideo(v, frame);
    %
    %    pause(1); % We can choose to pause to better understand the
    %                visualisation.
    %end
    %
    %hold off; close(v); % Close the video file
    %% Computing and Storing the SIFT descriptors
    % Why do we initialize the table of SIFT descriptors first with the
    % dimensions of y and then with the dimensions of x?
    %
    % The order of dimensions is a convention that depends on how the data
    % is accessed. Here, the data is created first on the y-axis and then
    % on the x-axis so that when we want to access sift_arr(y,x,:) we are
    % effectively accessing the SIFT descriptors at the (x,y) point of the
    % grid. We also note that the SIFT array has dimensions equal to the
    % size of the grid we want with the additional dimension of the 128
    % features we want.

    sift_arr=zeros([length(grid_y) length(grid_x) Options.Angles* ...
        Options.numBins*Options.numBins], ...
        'single');
    b = 0;

    % The initialization of the loop is done in such a way that it runs for
    % the total number of bins in each dimension of the patch for the SIFT
    % descriptors.
    for n = 1:Options.numBins*Options.numBins

        % Every 8 pixels, we place the data from the orientation image in
        % the third argument of sift_arr. Per iteration, we center on the
        % respective grid points and at the given coordinates we get the
        % orientation of the corners from the third dimension of the
        % orientation image matrix.
        sift_arr(:,:,b+1:b+Options.Angles) = I_orientation(grid_y+ ...
            sample_y(n), grid_x+sample_x(n), :);
        b = b+Options.Angles;
    end
    clear I_orientation


    % Outputs:
    [grid_x,grid_y] = meshgrid(grid_x, grid_y);
    [nrows, ncols, cols] = size(sift_arr);

    %% Normalizing the SIFT desriptors

    % Normalize SIFT descriptors slow, good normalization that respects the
    % flat areas

    % We form a global matrix, where the first dimension is the product of
    % the image grid dimensions and the second dimension is the number of
    % features.
    sift_arr = reshape(sift_arr, [nrows*ncols Options.Angles*Options.numBins* ...
        Options.numBins]);

    sift_arr = SIFTnormalizationVasilakis(sift_arr);  

    % We change the dimensions of the sift_arr table to a-by-b-by-128.
    sift_arr = reshape(sift_arr, [nrows ncols Options.Angles* ...
                                  Options.numBins*Options.numBins]);

    % slow bad normalization that does not respect the flat areas ct = .1;
    % sift_arr = sift_arr + ct; tmp = sqrt(sum(sift_arr.^2, 3)); sift_arr =
    % sift_arr ./ repmat(tmp, [1 1 size(sift_arr,3)]);

    % Transform the siftArr matrix from a-by-b-by-128 into a*b-by-128
    % dimensions
    sift_arr = reshape(sift_arr, ...
        [size(sift_arr,1)*size(sift_arr,2) size(sift_arr,3)]);
    
    features.data = sift_arr;
    features.x = grid_x(:);% + params.patchSize/2 - 0.5;
    features.y = grid_y(:);% + params.patchSize/2 - 0.5;
    features.wid = wid;
    features.hgt = hgt;

    feat_datastore{i,1} = arrayDatastore(features,"OutputType","same");
    %feat_tall_datastore = tall(feat_datastore);
    % whos feat_tall_datastore
    
    clear sift_arr features tmp 
    
    
end


features = feat_datastore;

end

% reset(grayTrainds); 
% reset(grayTestds);

