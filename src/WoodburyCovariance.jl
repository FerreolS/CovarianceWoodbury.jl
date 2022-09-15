using Zygote, OptimPackNextGen

struct WoodburyCovariance{T,N} <: Covariance{T,N}
	W :: AbstractArray{T,1} # Diagonal
	U :: AbstractArray{T,2} #
	denom :: AbstractArray{T}
	width :: NTuple
	rank :: Integer
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
		denom =  inv(I + U'*(W.*U))
		return new{T,N}(W,U,denom,width, rank)
	end

	function WoodburyCovariance{T}(width::Int,rank::Int) where {T<:Real}
		return WoodburyCovariance{T}(Tuple(width),rank::Int)
	end

	function WoodburyCovariance{T}(width::NTuple,rank::Int) where {T<:Real}
		N  = 2*length(width)
		L = prod(width)
		W = ones(T,L)
		U = zeros(T,L,rank)
		return new{T,N}(W,U,width, rank)
	end
	
	function WoodburyCovariance(width,rank::Int) 
		return WoodburyCovariance{Float64}(width, rank)
	end
end

function update!(A::WoodburyCovariance,W,U)
	A.W[:] .= W
	A.U[:,:] .= U
	A.denom .=  inv(I + U'*(W.*U))
	return A
end
function update!(A::WoodburyCovariance)
	A.denom .=  inv(I + A.U'*(A.W.*A.U))
	return A
end
Base.length(A::WoodburyCovariance) = prod(A.width).^2
Base.size(A::WoodburyCovariance) = (A.width..., A.width...)

function Base.getindex(A::WoodburyCovariance, I...)
	@assert iseven(length(I)) "Dimension must be even"
	L = length(I)/2
 	M = prod(I[1:L])
	N = prod(I[L+1:end])

	if M==N
		return 1/A.W[M] + A.U[M,:]'*A.U[N,:]
	else
		return A.U[M,:]'*A.U[N,:]
	end 
end


function Base.show(io::IO, obj::WoodburyCovariance{T,N}) where {T,N} 
    join(io, size(obj),"×")
    print(io, " WoodburyCovariance{$T,$N} with rank ", obj.rank)
end

	
function initialize!(A::WoodburyCovariance{T,N}) where {T,N}
	fill!(A.W,one(T))
	fill!(A.U,zero(T))
	update!(A)
	return A
end

function buildCovariance(data::AbstractArray{T},rank::Int) where {T<:Real}
	width = size(data)
	A = WoodburyCovariance{T}(width, rank)
	train!(A,data)
end

function ShermanLkl(r::M,W::V,U::M) where {T<:AbstractFloat,
	M<:AbstractMatrix{T},
	V<:AbstractVector{T}}
	L,N = size(r)
	
	ra = r.* W
	denom =  (I + U'*(W.*U))
	χ2 = sum(ra .* r , dims=1) .- sum(ra .* (U / denom * U' * ra), dims=1)
	Δa = sum(log,W)
	return 1/2 * (sum(χ2)/(L*N)  .- (Δa .- logdet(denom))/L)
end

function train!(A::WoodburyCovariance{T,N}, data::AbstractArray{T} ; verb=false,maxiter=1000) where {T,N}
	L = prod(A.width)
	bounds = zeros(T,L,A.rank+1)
	bounds[:,2:end] .=-Inf 
	xinit = hcat(A.W,A.U)
	
	cost(x)  = ShermanLkl(data,x[:,1],x[:,2:end]) 
	xsol = vmlmb(cost, xinit, lower=bounds,verb=verb,autodiff=true,maxiter=maxiter)
	update!(A, xsol[:,1], xsol[:,2:A.rank+1])
	return A
end

function apply(A::WoodburyCovariance,r::AbstractArray)
	ra = r.* A.W
	return ra .+ A.W*(A.U * A.denom * A.U' * ra)
end

function squarednorm(A::WoodburyCovariance, r::AbstractArray)
	ra = r.* A.W
	return ra .* (r .+  (A.U * A.denom * A.U' * ra))
end