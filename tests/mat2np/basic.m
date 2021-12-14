A=3;
disp(' a b " ');
disp('x');
disp(A(:, 4, 2:end));

clear A;

[out, ~] = func_example(A, A, A);

function [out, out2] = func_example(A, B, C)
    out = [];
    out = [1;2;3];
    out = [-.1 -.1 .1; 1, 2, 3];
    out2 = [out; out] / B;
    out2 = out{A, B};

    if (C == 1) && (B > 3)
        out2 = out2 + 1;
    elseif C == 2
    else
        for i = 1:A
            continue
            out2 = addone(out2(1:5:2), end) + out(i);
            break
        end
        return
    end

end


function out = addone(A)
    out=A+1;
end
