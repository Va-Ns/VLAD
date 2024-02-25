function sift_arr = SIFTnormalizationVasilakis(sift_arr)   

    %% Normalization process
    % normalize SIFT descriptors (after Lowe)

    % find indices of descriptors to be normalized (those whose norm is
    % larger than 1)

    % At this point we identify which of the descriptors should be
    % normalized. As a criterion for the latter, we set that we want their
    % norm to be greater than 1.

    % Great caution! The calculation of the norm is done in each row of the
    % sift_arr table because each row also reflects a separate keypoint in
    % the image.
    tmp = sqrt(sum(sift_arr.^2, 2));

    % We find those indicators that meet our criteria.
    normalize_ind = find(tmp > 1);

    % We select the SIFT data based on the indicators that meet the
    % criterion set above and place them in a new variable.
    sift_arr_norm = sift_arr(normalize_ind,:);

    % We perform element-wise division so that we can normalize the SIFT
    % descriptors
    sift_arr_norm = sift_arr_norm ./ repmat(tmp(normalize_ind,:),[1 size(sift_arr,2)]);
    %                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    %                                Here we repeat the norm table with all
    %                                those values that we found to be
    %                                greater than 1, as many times as the
    %                                second dimension of the variable
    %                                sift_arr.

    % suppress large gradients

    % We suppress large gradients. The reason why we do this is to reduce
    % the influence of high contrast on the image, such as edges. This is
    % because features with high contrast values can in the descriptor and
    % make it sensitive to changes or noise.
    sift_arr_norm(find(sift_arr_norm > 0.2)) = 0.2;

    % finally, renormalize to unit length

    % Here we will need to renormalize, because the process of attenuation
    % of large gradients has already taken place. So since the values of the
    % matrix have changed we will have to re-normalize.
    tmp = sqrt(sum(sift_arr_norm.^2, 2));
    sift_arr_norm = sift_arr_norm ./ repmat(tmp, [1 size(sift_arr,2)]);
    
    % This syntax is used to place only those pointers that need to be
    % normalized back into sift_arr. For this reason, we can also check
    % that the variable sift_arr_norm and the variable normalize_ind have
    % the same size in the first dimension.
    sift_arr(normalize_ind,:) = sift_arr_norm;
end
