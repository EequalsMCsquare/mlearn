subroutine dot_sum (result, x, w, length) bind(c)
    use iso_c_binding
    integer(c_int), intent(in) :: length
    real(c_double), dimension(length),intent(in) :: x
    real(c_double), dimension(length),intent(in) :: w 
    real(c_double), intent(out) :: result
    real(c_double) temp(length)

    temp = x * w
    result = sum(temp)
    return
end subroutine

