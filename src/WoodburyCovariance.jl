using Zygote, OptimPackNextGen, StatsBase, ArrayTools, TSVD

"""
	WoodburyCovariance{T,N} <: Covariance{T,N} <: AbstractArray{T,N}

is a covariance matrix stored in the form:

C = W^-1 + U C U'

where U is a low rank rectangular matrix. 

If the input space is of size `L` and `U` is of rank `K`, only `L*(K+1)+K^2` 
coefficients must be stored instead of at least `L(L+1)/2` for the full covariance matrix.

Under this form, applying this C or its inverse (i.e. precision matrix)
is trivial thanks to the matrix inversion lemma.

"""
struct WoodburyCovariance{T,N} <: Covariance{T,N}
	W :: AbstractArray{T,1} # Diagonal (size = width)
	U :: AbstractArray{T,2} # width x rank  matrix
	C :: Union{AbstractArray{T,2},LinearAlgebra.UniformScaling{Bool}}# rank x rank matrix
	denom :: AbstractArray{T} # 1/(C^-1 + U' W U)
	width :: NTuple			# size of the input space
	rank :: Integer			# rank of U
end


"""
    WoodburyCovariance(W::AbstractArray{T1} , U::AbstractArray{T2} ) where {T1<:Real,T2<:Real}

Build a WoodburyCovariance object from `W` and `U`. 
Its `width` is `size(W)` and its `rank` is `last(size(U))` .
`C` is the identity matrix.
"""
function WoodburyCovariance(W::AbstractArray{T1} , U::AbstractArray{T2} ) where {T1<:Real,T2<:Real}
	T = promote_type(T1,T2)
	W,U = convert(AbstractArray{T},W), convert(AbstractArray{T},U)
	width = size(W)
	N  = 2*length(width)
	if ndims(W)==ndims(U) && width == size(U) # add singleton dimension if needed
		U = reshape(U[:],Val(2))
		rank = 1
	elseif (ndims(W) +1) ==ndims(U)
		@assert width == size(U[..,1]) 
		rank = last(size(U))
		U = reshape(U,prod(width),rank)
	else
		error("U and W are not conformable")
	end
	W = W[:]
	C = I
	denom =  inv(C + U'*(W.*U))
	return WoodburyCovariance{T,N}(W,U,C,denom,width, rank)
end


"""
    WoodburyCovariance{T}(width,rank ) where {T<:Real}

Build a WoodburyCovariance object of size `width x width` and rank `rank`
"""
function WoodburyCovariance{T}(width::Int,rank::Int) where {T<:Real}
	return WoodburyCovariance{T}(Tuple(width),rank::Int)
end
function WoodburyCovariance(width,rank::Int) 
	return WoodburyCovariance{Float64}(width, rank)
end
function WoodburyCovariance{T}(width::NTuple,rank::Int) where {T<:Real}
	N  = 2*length(width)
	L = prod(width)
	W = ones(T,L)
	U = zeros(T,L,rank)
	denom = zeros(T,rank,rank)
	return WoodburyCovariance{T,N}(W,U,I,denom,width, rank)
end


function update!(A::WoodburyCovariance,W,U)
	A.W[:] .= W
	A.U[:,:] .= U
	A.denom .=  inv(inv(A.C) + U'*(W.*U))
	return A
end

#
# Basic operations on WoodburyCovariance
#
Base.length(A::WoodburyCovariance) = prod(A.width).^2
Base.size(A::WoodburyCovariance) = (A.width..., A.width...)

function Base.getindex(A::WoodburyCovariance{T, N}, I::CartesianIndex{N}) where {T,N}
	@assert iseven(N) "Dimension must be even"
	L = N/2
 	P = prod(I.I[1:L])
	Q = prod(I.I[L+1:end])

	if P==Q
		return 1/A.W[P] + A.U[P,:]' * A.C * A.U[Q,:]
	else
		return A.U[P,:]'* A.C * A.U[Q,:]
	end 
end

function Base.getindex(A::WoodburyCovariance, I...)
	@assert iseven(length(I)) "Dimension must be even"
	L = length(I)/2
 	M = prod(I[1:L])
	N = prod(I[L+1:end])

	if M==N
		return 1/A.W[M] + A.U[M,:]' * A.C * A.U[N,:]
	else
		return A.U[M,:]'* A.C * A.U[N,:]
	end 
end


function Base.show(io::IO, obj::WoodburyCovariance{T,N}) where {T,N} 
    join(io, size(obj),"×")
    print(io, " WoodburyCovariance{$T,$N} with rank ", obj.rank)
end


"""
    update!(A::WoodburyCovariance{T,N}) where {T,N}

Internal function to update cached variables 
"""
function update!(A::WoodburyCovariance{T,N}) where {T,N}
	A.denom .=  inv(inv(A.C) + A.U'*(A.W.*A.U)) 
	return A
end
	
"""
    reset!(A::WoodburyCovariance{T,N}) where {T,N}

reset the WoodburyCovariance to identity
"""
function reset!(A::WoodburyCovariance{T,N}) where {T,N}
	fill!(A.W,one(T))
	fill!(A.U,zero(T))
	update!(A)
	return A
end

"""
    buildCovariance(data::AbstractArray{T},rank::Int ; kwargs...) where {T<:Real}

Build the covariance matrix learned from the data. 
return the learned WoodburyCovariance 
"""
function buildCovariance(data::AbstractArray{T},rank::Int ; kwargs...) where {T<:Real}
	return buildCovariance(Val(:random),data,rank ; kwargs...) 
end

"""
    buildCovariance(data::AbstractArray{T},rank::Int ; kwargs...) where {T<:Real}

Build the covariance matrix learned from the data. 
return the learned WoodburyCovariance 
"""
function buildCovariance(::Val{:random},data::AbstractArray{T},rank::Int ; kwargs...) where {T<:Real}

	A = WoodburyCovariance(1 ./ var(data, dims=ndims(data))[..,1] , randn(size(data)[1:end-1]...,rank) );
	train!(A,data  ;  kwargs...)
end

"""
    buildCovariance(data::AbstractArray{T},rank::Int ; kwargs...) where {T<:Real}

Build the covariance matrix learned from the data. 
return the learned WoodburyCovariance 
"""
function buildCovariance(::Val{:svd},data::AbstractArray{T},rank::Int ; kwargs...) where {T<:Real}
	Cemp = cov(data,dims=2)
	U, s, V = tsvd(Cemp,rank)
	A = WoodburyCovariance(1 ./ var(data, dims=ndims(data))[..,1] , sqrt.(s').*V);
	train!(A,data  ;  kwargs...)
end


"""
    WoodburyLkl(r::M,W::V,U::M) where {T<:AbstractFloat,
										M<:AbstractMatrix{T},
										V<:AbstractVector{T}}

Internal cost function used for CovarianceWoodbury learning
"""
function WoodburyLkl(r::M,W::V,U::M) where {T<:AbstractFloat,
	M<:AbstractMatrix{T},
	V<:AbstractVector{T}}
	L,N = size(r)
	
	ra = r.* W
	denom =  (I + U'*(W.*U))
	χ2 = sum(ra .* r , dims=1) .- sum(ra .* (U / denom * U' * ra), dims=1)
	Δa = sum(log,W)
	return 1/2 * (sum(χ2)/(L*N)  .- (Δa .- logdet(denom))/L)
end

"""
    train!(A::WoodburyCovariance{T,N}, data::AbstractArray{T} ; kwargs...) where {T,N}

Train the WoodburyCovariance `A` matrix on  `data` 
"""
function train!(A::WoodburyCovariance{T,N}, data::AbstractArray{T} ; kwargs...) where {T,N}
	L = prod(A.width)
	bounds = zeros(T,L,A.rank+1)
	bounds[:,2:end] .=-Inf 
	xinit = hcat(A.W,A.U)
	
	cost(x)  = WoodburyLkl(data,x[:,1],x[:,2:end]) 
	xsol = vmlmb(cost, xinit; lower=bounds, autodiff=true , kwargs...)
	update!(A, xsol[:,1], xsol[:,2:A.rank+1])
	return A
end


"""
 Overloaded multiplication and inverse for WoodburyCovariance
"""
Base.:*(A::WoodburyCovariance{T, N},r::AbstractArray{T,S}) where {N,T<:Real,S} = apply(A,r)

Base.:*(A::WoodburyCovariance{T, 2},r::AbstractVector{T}) where {T<:Real} = apply(A,r)

function apply(A::WoodburyCovariance,r::AbstractArray)
	t = apply(A,r[:])
	return reshape(t,A.width) 
end

function apply(A::WoodburyCovariance,r::AbstractVector)
	return  r ./ A.W .+  (A.U * A.C * A.U' * r)
end

function Base.:\(A::WoodburyCovariance{T, N},r::AbstractArray{T,S}) where {N,T<:Real,S} 
	apply_inverse(A,r)
end

function Base.:\(A::WoodburyCovariance{T, N},r::AbstractArray{T,N}) where {N,T<:Real} 
	s = similar(r)
	ax = axes(r)
	for I in CartesianIndices(ax[N/2+1:end])
		t =  apply_inverse(A,r[ax[1:(N/2)]...,I][:])
		s[ax[1:(N/2)]...,I] .= reshape(t,A.width) 
	end
	return s
end

function Base.:\(A::WoodburyCovariance{T, 2}, r::AbstractMatrix{T}) where T<:Real
	s = similar(r)
	ax = axes(r)
	for I in CartesianIndices(ax[2])
		t =  apply_inverse(A,r[ax[1],I][:])
		s[ax[1],I] .= reshape(t,A.width) 
	end
	return s

end

Base.:\(A::WoodburyCovariance{T, 2},r::AbstractVector{T}) where {T<:Real} = apply_inverse(A,r)


function apply_inverse(A::WoodburyCovariance,r::AbstractArray)
	t = apply_inverse(A,r[:])
	return reshape(t,A.width) 
end

function apply_inverse(A::WoodburyCovariance,r::AbstractVector)
	ra = r.* A.W
	return ra .- A.W .* (A.U * A.denom * A.U' * ra)
end
