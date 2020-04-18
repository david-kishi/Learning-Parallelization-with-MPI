program main
    double precision :: a, b
    a = 0.0
    call sub(a, 1.0, b)
    print*, a, b
    end
    
    subroutine sub(i, j)
        integer :: i, j
        i=i+1
        j = 10.0
        end
    