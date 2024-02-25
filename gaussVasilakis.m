function [GX,GY] = gaussVasilakis(sigma)
        
        if all(size(sigma) == [1 1])

            % Isotropic Gaussian

            f_wid = 4 * ceil(sigma) + 1;
            G = fspecial("gaussian",f_wid,sigma);
            
            %	G = normpdf(-f_wid:f_wid,0,sigma);
            %	G = G' * G;
        
        else
           
            % anisotropic gaussian
            f_wid_x = 2 * ceil(sigma(1)) + 1;
            f_wid_y = 2 * ceil(sigma(2)) + 1;
            G_x = normpdf(-f_wid_x:f_wid_x,0,sigma(1));
            G_y = normpdf(-f_wid_y:f_wid_y,0,sigma(2));
            G = G_y' * G_x;

        end

        [GX,GY] = gradient(G);
        
        % Normalization of the GX and GY components
        GX = GX * 2 ./ sum(sum(abs(GX)));
        GY = GY * 2 ./ sum(sum(abs(GY)));
        
end