A=A+3+4i;
disp(' a b " ');
disp('x');
disp(A(:, 4, 2:end));

clear A;

[out(4), ~] = func_example(~A(1, end), A(end-1)', A(3:(end-1)).');

function [out, out2] = func_example(A, B, C)
    out = [];
    out = [1;2;3];
    out = [-.1 -0.1 .1; 1, func_example(A, B, C)+out+3, 3];
    out2 = [out; out] / B;
    out2 = out{A, B};

    if (C == 1) && (B > 3) || (A==4)
        out2 = out2 | 1;
    elseif C == 2
    else
        for i = 1:A
            continue
            out2 = addone(out2(1:5:2)./A, end) + out(i)\out(:);
            break
        end
        return
    end

end


function out = addone(A)
    out=A+1;
end
