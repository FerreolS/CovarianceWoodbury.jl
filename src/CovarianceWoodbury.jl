module CovarianceWoodbury

export ishermitian, WoodburyCovariance, 
	   initialize!, buildCovariance, train!,
	   apply, squarednorm

import LinearAlgebra: ishermitian

using LinearAlgebra, ArrayTools



abstract type Covariance{T,N} <: AbstractArray{T,N} end

Base.eltype(A::Covariance{T,N}) where {T,N} = T
Base.ndims(A::Covariance{T,N}) where {T,N} = N

LinearAlgebra.ishermitian(A::Covariance) = true

include("WoodburyCovariance.jl")


end # module CovarianceWoodbury
