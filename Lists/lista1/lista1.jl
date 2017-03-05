#---------------------Atividade 1---------------------
ex01(matrix) = 
    [ j == find( x->(x == maximum(matrix[i,:])), matrix[i,:] )[1] ? 1 : 0 
    for i = 1:size(matrix, 1), j = 1:size(matrix, 2) ]

a = rand(5, 3)

println(a)
println(ex01(a))

#---------------------Atividade 2---------------------
M = [5 10 -5 22; 1 33 15 3; 8 29 12 1; 3 11 39 20]

for i=1:3
    idx = find(x->x==maximum(M), M)
    M[idx] = 0
end

println(M)

#---------------------Atividade 3---------------------
function ex03(matrix)
    for c = 1: size(matrix, 2)
        matrix[:,c] = matrix[:,c] .- mean(matrix[:,c])
        matrix[:,c] = matrix[:,c] .* (1 / std(matrix[:,c]))
    end
end

a = rand(1.0:9.0, 5, 5)

println( a )
println( round(mean(a)) )
println( round( std(a)) )

ex03( a )

println( round(a) )
println( round(mean(a)) )
println( round( std(a)) )

#---------------------Atividade 4.1---------------------
using PyPlot

min = -2.0
max = 2.0

distribuicao = min + rand(500) .* (max - min)
distribuicao = sort(distribuicao)

seno = map(sin, distribuicao)
cosseno = map(cos, distribuicao)
cossecante = map(x-> (1 / sin(x)), distribuicao)
secante = map(x-> (1 / cos(x)), distribuicao)

xlim([min, max])

xlabel("Amostra")
ylabel("Métricas")

p1 = plot(distribuicao, seno, "r", linewidth=2)
p2 = plot(distribuicao, cosseno, "b", linewidth=2)

legend((p1[1], p2[1]), ("Seno", "Cosseno"))

#---------------------Atividade 4.2---------------------
p1 = plot(distribuicao, cossecante, "r", linewidth=2)
p2 = plot(distribuicao, secante, "b", linewidth=2)

legend((p1[1], p2[1]), ("Cossecante", "Secante"))

#---------------------Atividade 5---------------------
using PyPlot

mu = 20.0
sigma = 5.0

a = randn(100)
a = sort(a)

a = a .* (sigma / std(a))
a = a .+ mu

anorm = map(x->(1/sqrt(2*(sigma^2)*pi)*e^((-(x-mu)^2)/(2*(sigma^2)))), a)

xlabel("Distribuição")
p = plot(a, anorm, "r")

#---------------------Atividade 6---------------------
function f_cria_func_transf(n, m, r, x, y)
    middle = ( convert(Int64, round(Int64, n/2)), convert(Int64, round(Int64, m/2)) )
    return [ sqrt( (i - middle[1]) ^ 2 + ( j - middle[2] ) ^ 2 ) <= r ? y : x 
                for i=1:n, j=1:m ]
end

print(f_cria_func_transf(7, 7, 2, 0, 1))