### A Pluto.jl notebook ###
# v0.12.2

using Markdown
using InteractiveUtils

# ╔═╡ d0aa7170-128a-11eb-0d83-4fdb51b995c4
using Random, Distributions,Plots,PyPlot

# ╔═╡ 2e117f52-127e-11eb-0d5d-7f1a5ccdbc36
function sigmoide(z)
	
	return 1/(1 + exp(-z))
	
end

# ╔═╡ ff454b4e-127f-11eb-396b-9964c9d2c684
begin 
	a= [ 0  1 2  -1] 
	b = sigmoide.(a)
	b
end 


# ╔═╡ 37b16aa0-1280-11eb-3692-dd46ad2ba110
function regresion_logistica(X, theta)
	return sigmoide.(X*theta)
	
end

# ╔═╡ 7b576110-1280-11eb-2308-1970134e411e
function costo(y_hat, y )
	m = size(y)[1]
	J = -(1.0/m) *((y' * log.(y_hat)) .+ (1 .- y')*log.(1 .- y_hat))
	
	return J[1]
	
end

# ╔═╡ 61b140a0-13ca-11eb-2895-8b89e1f6e64f
function descenso_gradiente(X,y, theta, alpha, iteraciones)
	
	#Número de ejemplos
	m = size(X)[1]
	J=0
	
	for i=1:iteraciones
		# Predicciones
		y_hat = regresion_logistica(X, theta)
		
		J = costo(y_hat,y)
		
		# Optimiza theta aplicando descenso por el gradiente
		theta = theta - alpha *(1.0/m) * X' *(y_hat .- y)
		
		end
	
	return J, theta
	
	end

# ╔═╡ 200c0d20-13d3-11eb-2413-d5d8c54d8b43
function predice_clase(X, theta)
	"""
	X: matrix de características (m,d+1)
	theta: vector de paramétros (d+1,)
	regresa una clase por cada ejemplo ( 0 o 1 )
	"""
	y_hat = regresion_logistica(X, theta)
	
	# Convierte las probabilidades en clase
	y_hat[ (1 .- y_hat) .< 0.5] .= 1 
	y_hat[y_hat .< 0.5] .=0 
	y_hat[y_hat .== 0.5] .= 0
	
	return y_hat
	
	
end

# ╔═╡ 7b1a09f0-1280-11eb-15b6-254fe407e6b2
function prueba(X,y,theta)
	m = size(X,1)
	
	y_hat = predice_clase(X,theta)
	
	precision = 1.0/m * sum(y_hat .== y)
	
	return precision 
end

# ╔═╡ c2dcb38e-128b-11eb-0f35-698048c7a6d9
begin 
	import Pkg; Pkg.add("Distributions")
	Pkg.add("PyPlot")
end

# ╔═╡ 7ae90ee0-1280-11eb-0089-bbfbb8902cfd
function datos(m)
	
	# m Número de ejemplos
	d = 3 #Número de dimensiones por ejemplo
	
	rng_pos = Normal(-2,1)
	rng_neg = Normal(1,2)
	
	pos =  rand(rng_pos,(m,d))
	clase_pos= ones(m,1)
	# Agrega la etiquetas como la última columna
	pos = [pos clase_pos]
	
	
	neg = rand(rng_neg, (m,d))
	clase_neg = zeros(m,1)
	# Agrega la etiquetas como la última columna
	neg = [neg clase_neg]
	
	
	X = [pos; neg]
	
	# Vector de sesgo 
	sesgo = ones(2*m,1)
	
	# Agrega el término de sesgo 
	X = [sesgo  X]
	
	# Mezcla los ejemplos  (los renglones)
	X = X[shuffle(1:end), :]
	
	# Obten las etiquetas
	y = X[:,end]
	
	# Toma solo las características
	X = X[:,1:size(X)[2]-1]
	
	
	
	return X,y
	
	
end 

# ╔═╡ 71318570-1295-11eb-3624-794a80021f97
begin 
	X,y =  datos(500)
	size(X),size(y)
end

# ╔═╡ 64bad6a2-1297-11eb-3b90-0bca2fe5321f
function visualiza_datos(X,y)
	# Selecciona solo los ejemplos positivos
	mask_pos = y .== 1.0
	pos_X = @view X[mask_pos,:]
	
	# Selecciona solo los ejemplos negativos
	mask_neg = y .!= 1.0
	neg_X  = @view X[mask_neg,:]
	
	close("all")
	title("Datos")
	using3D()
	
	fig = figure()
	subplot(111, projection="3d")
	ax = Axes3D(fig)
	# Ejemplos positivos "1"
	ax.scatter(pos_X[:,2],pos_X[:,3],pos_X[:,4],marker = "^", color = "blue")
	# Ejemplos positivos "0"
	ax.scatter(neg_X[:,2],neg_X[:,3],neg_X[:,4],marker = "o", color = "red")
	gcf()
	
end

# ╔═╡ 3f94fdf0-1298-11eb-0d6d-25780b4d6e74
visualiza_datos(X,y)

# ╔═╡ 32fba710-13ba-11eb-1d05-a77c281617bd
begin 
	m,d = size(X)
	fracc = 0.1 # fracción conjunto de entrenamiento
	n_ent = Integer(m *fracc)
	X_ent=  @view X[1:n_ent,:]
	 
	X_prueba = @view X[n_ent + 1:end,:]
	y_ent = @view y[1:n_ent]
	y_prueba = @view y[n_ent + 1:end]
	
	#Inicializa el vector de parámetros
	theta = zeros(d,1)
	alpha = 0.3
	iteraciones = 10 
	
	
	#Optimiza theta con respecto de J (función de costo)
	J, theta =  descenso_gradiente(X,y,theta, alpha, iteraciones)
	
	size(y_prueba)
end 

# ╔═╡ a8313740-13b8-11eb-134c-599aba47eba3
J, theta

# ╔═╡ 34322b40-13ba-11eb-0991-c109c50d26eb
prueba(X_prueba, y_prueba, theta)

# ╔═╡ Cell order:
# ╠═2e117f52-127e-11eb-0d5d-7f1a5ccdbc36
# ╠═ff454b4e-127f-11eb-396b-9964c9d2c684
# ╠═37b16aa0-1280-11eb-3692-dd46ad2ba110
# ╠═7b576110-1280-11eb-2308-1970134e411e
# ╠═61b140a0-13ca-11eb-2895-8b89e1f6e64f
# ╠═200c0d20-13d3-11eb-2413-d5d8c54d8b43
# ╠═7b1a09f0-1280-11eb-15b6-254fe407e6b2
# ╠═c2dcb38e-128b-11eb-0f35-698048c7a6d9
# ╠═d0aa7170-128a-11eb-0d83-4fdb51b995c4
# ╠═7ae90ee0-1280-11eb-0089-bbfbb8902cfd
# ╠═71318570-1295-11eb-3624-794a80021f97
# ╠═64bad6a2-1297-11eb-3b90-0bca2fe5321f
# ╠═3f94fdf0-1298-11eb-0d6d-25780b4d6e74
# ╠═32fba710-13ba-11eb-1d05-a77c281617bd
# ╠═a8313740-13b8-11eb-134c-599aba47eba3
# ╠═34322b40-13ba-11eb-0991-c109c50d26eb
