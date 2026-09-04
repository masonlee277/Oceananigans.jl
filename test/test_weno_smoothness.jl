include("dependencies_for_runtests.jl")

using Oceananigans.Advection: beta_loop, biased_weno_weights

@testset "Float32 WENO smoothness indicators" begin
    # A smooth ρθ profile with large mean (≈ 300) and small O(0.1) perturbations.
    # This is the scenario that triggers catastrophic cancellation in naive Float32
    # β computation: the quadratic form accumulates terms ~ 300² ≈ 9e4 that must
    # cancel to leave a residual ~ 0.01, exceeding Float32's ~7-digit precision.
    for order in (5, 7, 9)
        buffer = Int((order + 1) ÷ 2)
        n_stencil = 2 * buffer  # full stencil width

        # Sample a smooth sinusoidal field: ρθ(x) = 300 + 0.1 sin(2π x)
        S_f64 = ntuple(i -> 300.0 + 0.1 * sinpi(2 * (i - 1) / n_stencil), n_stencil)
        S_f32 = ntuple(i -> Float32(S_f64[i]), n_stencil)

        # Build sub-stencils (left-bias ordering, matching S₀ₙ … S₍ₙ₋₁₎ₙ)
        ψ_f64 = ntuple(Val(buffer)) do k
            start = buffer - k + 1
            ntuple(j -> S_f64[start + j - 1], Val(buffer))
        end

        ψ_f32 = ntuple(Val(buffer)) do k
            start = buffer - k + 1
            ntuple(j -> S_f32[start + j - 1], Val(buffer))
        end

        scheme_f64 = WENO(Float64; order, weight_computation=Oceananigans.Utils.NormalDivision)
        scheme_f32 = WENO(Float32; order, weight_computation=Oceananigans.Utils.NormalDivision)

        β_f64 = beta_loop(scheme_f64, ψ_f64)
        β_f32 = beta_loop(scheme_f32, ψ_f32)

        @info "WENO order $order β (Float64): $β_f64"
        @info "WENO order $order β (Float32): $β_f32"

        @testset "WENO order $order" begin
            # All Float32 β values must be non-negative
            # (negative β was the symptom of catastrophic cancellation)
            for r in 1:buffer
                @test β_f32[r] >= 0
            end

            # Float32 β should approximate Float64 reference
            for r in 1:buffer
                if β_f64[r] > 0
                    @test β_f32[r] ≈ β_f64[r] rtol=1e-2
                end
            end

            # Weights must sum to 1 and match Float64 reference
            ω_f64 = biased_weno_weights(ψ_f64, nothing, scheme_f64)
            ω_f32 = biased_weno_weights(ψ_f32, nothing, scheme_f32)

            @test sum(ω_f64) ≈ 1
            @test sum(ω_f32) ≈ 1

            for r in 1:buffer
                @test ω_f32[r] ≈ ω_f64[r] atol=1e-3
            end
        end
    end
end

@testset "Float32 WENO weights beside a large jump" begin
    # A tracer that is exactly zero on one side of a front and of magnitude ~1e8 on the other
    # (an ice number concentration in kg⁻¹ at a cloud edge). The sub-stencil inside the zero
    # region is exactly flat (β = 0), so its ratio τ / (β + ϵ) ~ ψ² / ϵ is of order 1e24, whose
    # square is beyond floatmax(Float32). The normalized evaluation must stay finite and agree
    # with the Float64 weights; the smooth-stencil weights must be unchanged by the normalization.
    for order in (3, 5, 7, 9), amplitude in (1e6, 1e8)
        buffer = Int((order + 1) ÷ 2)
        n_stencil = 2 * buffer

        S_f64 = ntuple(i -> i > buffer + 1 ? amplitude : i == buffer + 1 ? amplitude / 2000 : 0.0, n_stencil)
        S_f32 = ntuple(i -> Float32(S_f64[i]), n_stencil)

        ψ_f64 = ntuple(Val(buffer)) do k
            start = buffer - k + 1
            ntuple(j -> S_f64[start + j - 1], Val(buffer))
        end

        ψ_f32 = ntuple(Val(buffer)) do k
            start = buffer - k + 1
            ntuple(j -> S_f32[start + j - 1], Val(buffer))
        end

        scheme_f64 = WENO(Float64; order, weight_computation=Oceananigans.Utils.NormalDivision)
        scheme_f32 = WENO(Float32; order, weight_computation=Oceananigans.Utils.NormalDivision)

        ω_f64 = biased_weno_weights(ψ_f64, nothing, scheme_f64)
        ω_f32 = biased_weno_weights(ψ_f32, nothing, scheme_f32)

        @testset "WENO order $order, amplitude $amplitude" begin
            @test all(isfinite, ω_f32)
            @test sum(ω_f64) ≈ 1
            @test sum(ω_f32) ≈ 1

            for r in 1:buffer
                @test ω_f32[r] ≈ ω_f64[r] atol=1e-3
            end
        end
    end

    # Ratios at most one: the normalized evaluation reproduces the textbook formula exactly.
    for order in (5, 9)
        buffer = Int((order + 1) ÷ 2)
        n_stencil = 2 * buffer
        S = ntuple(i -> 300.0f0 + 0.1f0 * sinpi(2f0 * (i - 1) / n_stencil), n_stencil)
        ψ = ntuple(Val(buffer)) do k
            start = buffer - k + 1
            ntuple(j -> S[start + j - 1], Val(buffer))
        end

        scheme = WENO(Float32; order, weight_computation=Oceananigans.Utils.NormalDivision)
        β = beta_loop(scheme, ψ)
        τ = Oceananigans.Advection.global_smoothness_indicator(Val(buffer), β)
        ϵ = Oceananigans.Advection.ϵ
        r = ntuple(s -> τ / (β[s] + ϵ), buffer)
        @test maximum(r) <= 1
        α = ntuple(s -> Oceananigans.Advection.C★(scheme, Val(s - 1)) * (1 + r[s]^2), buffer)
        @test Oceananigans.Advection.zweno_alpha_loop(scheme, β, τ) == α
    end
end

@testset "Bounds-preserving scaling at the bound" begin
    using Oceananigans.Advection: bounds_preserving_scaling

    # A Float32 tracer at its lower bound whose face reconstructions sit at the 1e-20 scale of a
    # microphysics floor: the old ε-regularized denominator m - c + 1e-20 cancelled exactly and
    # returned 0 / 0. The scaling must be 0 here (the faces collapse to the cell value).
    θ = bounds_preserving_scaling(0f0, 2.2f-20, -1f-20, 0f0, 1f0)
    @test isfinite(θ)
    @test θ == 0

    # No scaling when both reconstructions equal the cell value.
    @test bounds_preserving_scaling(0.3f0, 0.3f0, 0.3f0, 0f0, 1f0) == 1

    # Random stencils: θ is finite, in [0, 1], and the scaled faces stay within the bounds.
    for FT in (Float32, Float64), trial in 1:2000
        c = rand(FT)
        c₋ᴿ = c + (rand(FT) - FT(0.5)) * FT(4)
        c₊ᴸ = c + (rand(FT) - FT(0.5)) * FT(4)
        θ = bounds_preserving_scaling(c, c₋ᴿ, c₊ᴸ, zero(FT), one(FT))
        @test isfinite(θ) && 0 <= θ <= 1
        tol = 8 * eps(FT)
        @test -tol <= θ * (c₋ᴿ - c) + c <= 1 + tol
        @test -tol <= θ * (c₊ᴸ - c) + c <= 1 + tol
    end
end
