program main
integer, parameter :: n = 10, lda = n, & 
ldb = n, nrhs = 1 
integer :: info, row, col, ipiv(n) 
double precision :: A(lda, n), b(ldb) 
do col = 1, n 
do row = 1, n 
A(row, col) = row - col
end do
A(col, col) = 1.0d0
b(col) = 1 + ( n * (2 * col - n - 1)) / 2 
end do 

call dgesv ( n, nrhs, A, lda, ipiv, b, ldb, info ) 
if ( info == 0 ) then 
print*, "Maximum error = ", maxval(abs(b - 1.0d0)) else 
print*, "Error in dgesv: info = ", info 
end if 
end program main 

