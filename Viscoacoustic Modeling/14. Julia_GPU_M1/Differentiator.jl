



struct Differentiator
    l::Int32
    lmax::Int32
    coeffs::Array{Float32,2}
    w::Array{Float32,1}
end

function Differentiator(l::Int32)
    lmax = 8
    if l < 1
        l = 1
    end
    if l > lmax
        l = lmax
    end

    coeffs = zeros(Float32, lmax, lmax)
    w = zeros(Float32, l)

    coeffs[1, 1] = 1.0021
    coeffs[2, 1:2] .= [1.1452, -0.0492]
    coeffs[3, 1:3] .= [1.2036, -0.0833, 0.0097]
    coeffs[4, 1:4] .= [1.2316, -0.1041, 0.0206, -0.0035]
    coeffs[5, 1:5] .= [1.2463, -0.1163, 0.0290, -0.0080, 0.0018]
    coeffs[6, 1:6] .= [1.2542, -0.1213, 0.0344, -0.0170, 0.0038, -0.0011]
    coeffs[7, 1:7] .= [1.2593, -0.1280, 0.0384, -0.0147, 0.0059, -0.0022, 0.0007]
    coeffs[8, 1:8] .= [1.2626, -0.1312, 0.0412, -0.0170, 0.0076, -0.0034, 0.0014, -0.0005]

    for k in 1:l
        w[k] = coeffs[l, k]
    end

    return Differentiator(l, lmax, coeffs, w)
end


function DiffDxminus(Diff::Differentiator, A::Array{Float32,2}, dA::Array{Float32,2}, dx::Float32)
    nx, ny = size(A)
    l = Diff.l
    w = Diff.w

    #A = OffsetArray(A, 0:nx-1, 0:ny-1) # 5x5 array with 0-based indices
    #A = OffsetArray(dA, 0:nx-1, 0:ny-1) 
    #l = OffsetArray(l, 0:nx-1, 0:ny-1) 
    
    for i in 1:l
        for j in 1:ny
            # Calculate the weighted sum for left border elements
            sum = 0.0
            for k in 1:i
                if i - k >= 1
                    sum -= w[k] * A[i - k, j]
                end

                #sum -= w[k] * A[i - k + 1, j]
            end
            for k in 1:l
                sum += w[k] * A[i + (k - 1), j]
            end
            dA[i, j] = sum / dx
        end
    end

    # Outside border area (l + 1 <= i < nx - l + 1)
    for i in (l+1):(nx - l)
        for j in 1:ny
            # Calculate the weighted sum for elements outside the border area
            sum = 0.0
            for k in 1:l
                sum += w[k] * (-A[i - k, j] + A[i + (k - 1), j])
            end
            dA[i, j] = sum / dx
        end
    end

    # Right border (nx - l + 1 <= i <= nx)
    for i in (nx - l + 1):nx
        for j in 1:ny
            # Calculate the weighted sum for right border elements
            sum = 0.0
            for k in 1:l
                sum -= w[k] * A[i - k, j]
            end
            for k in 1:(nx - i)
                sum += w[k] * A[i + (k - 1), j]
            end
            dA[i, j] = sum / dx
        end
    end
end
    

function DiffDxplus(Diff::Differentiator, A::Array{Float32,2}, dA::Array{Float32,2}, dx::Float32)
    nx, ny = size(A)
    l = Diff.l
    w = Diff.w

    # Left border (1 < i < l + 1)
    for i in 1:l
        for j in 1:ny
            sum = 0.0
            for k in 1:i
                sum -= w[k] * A[i - (k - 1), j]
            end
            for k in 1:l
                sum += w[k] * A[i + k, j]
            end
            dA[i, j] = sum / dx
        end
    end

    # Between left and right border
    for i in (l+1):(nx - l)
        for j in 1:ny
            sum = 0.0
            for k in 1:l
                sum += w[k] * (-A[i - (k - 1), j] + A[i + k, j])
            end
            dA[i, j] = sum / dx
        end
    end

    # Right border
    for i in (nx - l + 1):nx
        for j in 1:ny
            sum = 0.0
            for k in 1:l
                sum -= w[k] * A[i - (k - 1), j]
            end
            for k in 1:(nx - i)
                sum += w[k] * A[i + k, j]
            end
            dA[i, j] = sum / dx
        end
    end
end
 

function DiffDyminus(Diff, A, dA, dx)
    nx, ny = size(A)
    l = Diff.l
    w = Diff.w

    for i in 1:nx
        for j in 1:l
            sum = 0.0
            for k in 1:j
                if j - k >= 1
                    sum -= w[k] * A[i, j - k]
                end
            end
            for k in 1:l
                sum += w[k] * A[i, j + (k - 1)]
            end
            dA[i, j] = sum / dx
        end
    end
    

    # Outside border area
    for i in 1:nx
        for j in (l+1):(ny - l)
            sum = 0.0
            for k in 1:l
                sum += w[k] * (-A[i, j - k] + A[i, j + (k - 1)])
            end
            dA[i, j] = sum / dx
        end
    end

    # Bottom border
    for i in 1:nx
        for j in (ny - l + 1):ny
            sum = 0.0
            for k in 1:l
                sum -= w[k] * A[i, j - k]
            end
            for k in 1:(ny - j)
                sum += w[k] * A[i, j + (k - 1)]
            end
            dA[i, j] = sum / dx
        end
    end
end

function DiffDyplus(Diff, A, dA, dx)
    nx, ny = size(A)
    l = Diff.l
    w = Diff.w

    # Top border (1 < j < l + 1)
    for i in 1:nx
        for j in 1:l
            sum = 0.0
            for k in 1:j
                sum -= w[k] * A[i, j - (k - 1)]
            end
            for k in 1:l
                sum += w[k] * A[i, j + k]
            end
            dA[i, j] = sum / dx
        end
    end

    # Outside border area
    for i in 1:nx
        for j in (l+1):(ny - l)
            sum = 0.0
            for k in 1:l
                sum += w[k] * (-A[i, j - (k - 1)] + A[i, j + k])
            end
            dA[i, j] = sum / dx
        end
    end

    # Bottom border
    for i in 1:nx
        for j in (ny - l + 1):ny
            sum = 0.0
            for k in 1:l
                sum -= w[k] * A[i, j - (k - 1)]
            end
            for k in 1:(ny - j)
                sum += w[k] * A[i, j + k]
            end
            dA[i, j] = sum / dx
        end
    end
end



