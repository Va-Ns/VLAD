function output = mustBePowerOfTwo(input)
    %% Validate that the value is a power of two
    % A number is a power of two if and only if it is greater than zero and
    % its bitwise AND with its predecessor is zero.
    assert(isscalar(input) && input > 0 && floor(input) == input, ...
        'Input must be a positive integer');
    assert(bitand(input, input - 1) == 0, 'Input is not a power of two');
    
end