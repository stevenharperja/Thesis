addpath(pwd)
function r = RankNorm(X) %Seems to work well.
  b = trace(X'*X);  % each weight to the second, summed.
  a = trace((X'*X)' * (X'*X)); %each weight to the 4th, summed.
  r = a / (b^2);
end


#{
M = [ 1, 2, 3, 4;
      5, 6, 7, 8;
      9,10,11,12;
     13,14,15,16];
#}
#
M = [  1,0.5,  0,  0;
       0,  0,  0,0.5;
       0,0.5,  0,0.5;
       0,  0,  1,  0]
#
M = M*2

[u,s,v] = svd(M);
disp(s);

% create rank 3 approximation
s1 = s * [1,0,0,0;0,1,0,0;0,0,1,0;0,0,0,0];
N = u * s1 * v';

% rank 2
s2 = s * [1,0,0,0;0,1,0,0;0,0,0,0;0,0,0,0];
O = u * s2 * v';

% rank 1
s3 = s * [1,0,0,0;0,0,0,0;0,0,0,0;0,0,0,0];
P = u * s3 * v';

disp("Increasingly low rank")
disp(RankNorm(M));
disp(RankNorm(N));
disp(RankNorm(O));
disp(RankNorm(P)); %higher as we go down the list means good.

disp("Removing each possible s value")

for i = 1:16
  dec2bin(
  sA = s *
  A = u * sA * v';
  disp(RankNorm(A));
endfor
