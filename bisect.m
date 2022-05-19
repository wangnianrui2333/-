%二分法
function [x,fx,iter,X]=bisect(fun,a,b,eps,varargin)
%二分法求解非线性方程的根

fa=feval(fun,a,varargin{:});
fb=feval(fun,b,varargin{:});

k=1;

if fa*fb>0
   warning(['区间[',num2str(a),',',num2str(b),']内可能没有根']);
elseif fa==0
    x=a;
    fx=fa;
elseif fb==0
    x=b;
    fx=fb;
else
    while abs(b-a)>esp
        x=(a+b)/2;
        fx=feval(fun,x,varargin{:});%计算中点函数值
        if fa*fx>0
           a=x;
           fa=fx;
        elseif fb*fx>0
            b=x;
            fb=fx;
        else
            break
        end
        X(k)=x;
        k=k+1;
    end
end
iter=k;
end