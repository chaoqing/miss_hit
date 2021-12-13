A;
3;

function [out, out2] = func_example(A, B, C)
    out = [];
    out = [1;2;3];
    out = [-.1 -.1 .1; 1, 2, 3];
    out2 = [out; out] / B;

    if C == 1
        out2 = out2 + 1;
    else
        for i = 1:A
            out2 = addone(out2) + out(i);
        end

    end

end


function out = addone(A)
    out=A+1;
end
