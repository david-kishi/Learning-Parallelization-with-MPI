!------------------------------------------------------------------------------
! FILE: fortran_matrix.f90
! DESCRIPTION:
!   A simple fortran program of basic matrice computations.
! AUTHOR: David Nguyen
! CONTACT: david@knytes.com
! LAST REVISED: 20/04/2020
!------------------------------------------------------------------------------

!------------------------------------------------------------------------------
! Main Program
!   - Declare a 4x4 matrix A. Set all the values in A to 4.
!   - Declare a 4x4 matrix B. Set B(i,j) = i+j+1
!   - Compute C = A + B
!   - Replace in C the elements with value 8 with the value 16.
!   - Displays A, B, and C.
!------------------------------------------------------------------------------
program main
    implicit none

    integer, parameter :: n = 4
    integer :: i, j
    integer, dimension(n,n) :: a, b, c

    ! Set all values in matrice A to 4
    a = 4

    ! Set values of matrice B to i + j + 1 and display
    do i=1,n
        do j=1,n 
            B(i,j) = i + j + 1
        end do
    end do

    ! Compute C = A + B
    do i=1,n
        do j=1,n 
            C(j,i) = A(j,i) + B(j,i)

            ! If C(j,i) == 8, replace with 16
            if(C(j,i) == 8) then
                C(j,i) = 16
            end if
        end do
    end do

    ! Print Matrices
    print *, "--- Matrix A ---"
    call printMatrix(n,n,A)
    print *, "--- Matrix B ---"
    call printMatrix(n,n,B)
    print *, "--- Matrix C ---"
    call printMatrix(n,n,C)

end program main

!------------------------------------------------------------------------------
! SUBROUTINE: printMatrix
! PARAMETERS:
!   - i_: # of rows
!   - j_: # of columns
!   - mat_: Matrix
! DESCRIPTION: Pretty prints matrix.
!------------------------------------------------------------------------------
subroutine printMatrix(i_, j_, mat_)
    implicit none

    integer :: i,j
    integer, intent(in):: i_,j_
    integer, intent(in), dimension(i_,j_) :: mat_

    do i=1,i_
        do j=1,j_
            write(*, "(I8)", ADVANCE="NO") mat_(i,j)
        end do
        print *
    end do
end subroutine printMatrix
