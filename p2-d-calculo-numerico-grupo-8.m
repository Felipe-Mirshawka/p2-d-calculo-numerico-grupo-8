clear all
close all
clc

columns_names = {'Sunshine hours',	'Cost bottle of water',	...
                 'Obesity Levels',	'Life expectancy(years)', ...
                 'Pollution(Index score)',	'Annual avg. hours worked',...
                 'Happiness levels(Country)',	'Outdoor activities',...
                 'No. of take out places',	'Monthly gym membership', 'HL_index'};

data = csvread('C:\Users\faica\Downloads\p2_dados.csv');
## removendo a linha com strings dos títulos
data(1, :) = [];
## removendo a coluna com strings das cidades
data(:, 1) = [];

"Data importado."

[n_rows, n_cols] = size(data);

# Loop para percorrer todas as colunas, exceto a 7ª
for col = 1:n_cols
    if col == 7
        continue;  # Ignora a 7ª coluna
    endif

    # Substituir 0 por NaN na coluna atual
    coluna = data(:, col);
    coluna(coluna == 0) = NaN;
    data(:, col) = coluna;
endfor

media_cols = mean(data, 'omitnan');

for col = 1:n_cols
    if col == 7
        continue;  # Ignora a 7ª coluna
    endif

    # Substituir NaN pela média na coluna atual
    coluna = data(:, col);
    coluna(isnan(coluna)) = media_cols(col);
    data(:, col) = coluna;
endfor

"NaNs corrigidos."

# Calcular a média e o desvio padrão de cada coluna
vec_med = mean(data);
vec_desvpad = std(data);

# Identificar outliers e removê-los
outliers = abs((data - vec_med) ./ vec_desvpad) > 3; # 3 é o max de desvios padrões antes de ser considerado outlier
data_sem_outliers = data(~any(outliers, 2), :);

"Outliers deletados."

# Calcular o valor mínimo e máximo de cada coluna
min_vals = min(data);
max_vals = max(data);

# Aplicar a normalização Min-Max
data_normalizado = (data - min_vals) ./ (max_vals - min_vals);

"Data normalizado."

 Separar os dados em dois dataframes: df1 e df0
df1 = data_normalizado(data_normalizado(:, 7) == 1, :);  # Dados onde Happiness Levels == 1
df0 = data_normalizado(data_normalizado(:, 7) == 0, :);  # Dados onde Happiness Levels == 0

"Data separado em DF0 e DF1."

figure (1);

for ii = 1:size(data_normalizado,2)
subplot(3,4,ii);
hold on
plot(1,df0(:,ii),'ro','MarkerSize',2);
plot(1,df1(:,ii),'bo','MarkerSize',2);
##boxplot(data_normalizado(:, 1:end, 'o'));
ylabel(columns_names(ii));
xtickangle(45)
endfor
hold off

# Gerar boxplot para df1 (Happiness Levels == 1)
figure (2);
hold on
grid on
boxplot(df1(:, 1:end));
title('Boxplot para df1 (Happiness Levels = 1)');
xlabel('Variáveis');
ylabel('Valores Normalizados');
xtickangle(45)
hold off


# Gerar boxplot para df0 (Happiness Levels == 0)
figure(3);
hold on
grid on
boxplot(df0(:, 1:end));
title('Boxplot para df0 (Happiness Levels = 0)');
xlabel('Variáveis');
ylabel('Valores Normalizados');
hold off



figure(4);
hold on
grid on

x = data_normalizado(:,4);
y = data_normalizado(:,7);
plot(x,y,'o');

p = polyfit(x,y,1);
y_fit = polyval(p,x);
plot(x,y_fit,'-r');

SS_tot = sum((y - mean(y)).^2);  % Soma total dos quadrados (SST)
SS_res = sum((y - y_fit).^2);    % Soma dos quadrados dos resíduos (SSR)
R2 = 1 - (SS_res / SS_tot)       % R²

title('Scatter entre HL_index e life expectancy');
xlabel('Life Expectancy');
ylabel('Happiness Levels');
hold off

x = data_normalizado(:, 1:end);
y = data(:, 7);  % Última coluna (happiness score)

# Calcular a correlação de Pearson para cada variável com o happiness score
[n_rows, n_cols] = size(x);
correlacoes = zeros(1, n_cols);

for i = 1:n_cols
    correlacoes(i) = corr(x(:, i), y);
endfor

# Exibir os coeficientes de correlação
disp('Coeficiente de Correlação de Pearson para cada variável:');
disp(correlacoes);


resp1 = ['Análise 1: Seleção das Variáveis'
        '1.1:  Quais 2 variáveis tem maior correlação com happiness level (HL) ALTO?'
        'R: Custo da garrafa de água e Expectativa de Vida'
        '1.2: Quais 2 variáveis tem maior correlação com happiness level (HL) BAIXO?'
        'R: Índice de poluição e Média de horas anuais trabalhadas'
        '1.3: Justifique (1.1) e (1.2) baseado em análises numéricas'
        'R: Índice de correlação de Pearson (IP):'
        '   IP(CustoÁgua) = 0.62017'
        '   IP(ExpecVida) = 0.4523'
        '   IP(Poluição) = -0.73652'
        '   IP(HorasTrabalho) = -0.49395']

####
####
####
####
####
####
####
####
####
####
####
####

#### Exercício 2

  2.1

figure(2);
for ii = 1:size(data_normalizado,2)
  subplot(3,4,ii);
  hold on
  plot(df0(:,ii),df0(:,11),'ro','MarkerSize',2);
  ylabel(columns_names(ii))
endfor
hold off

## Resp1: Custo da garrafa de agua

######  2.2

## MÉTODO 1

# Variável independente (Life Expectancy)
x = df0(:, 2);

# Variável dependente transformada (logaritmo dos níveis de felicidade)
log_y = log(df0(:, 7));

# Montar matriz de design (X) e o vetor de resposta (log_y)
X = [ones(length(x), 1), x];

# Decomposição LU
[L, U] = lu(X' * X);
log_y_transformed = X' * log_y;

# Resolução de Ly = b usando substituição direta
y_temp = linsolve(L, log_y_transformed);

# Resolução de Ux = y usando substituição retroativa
b = linsolve(U, y_temp);

# Transformar de volta para a forma exponencial
a = exp(b(1));
b_exp = b(2);

# Criar um vetor denso para x, para suavizar a linha de ajuste
x_dense = linspace(min(x), max(x), 100);

# Prever novos valores para x_dense
y_pred_dense = a * exp(b_exp * x_dense);

# Calcular valores previstos para x
y_pred = a * exp(b_exp * x);

# Calcular os resíduos
residuals = df0(:, 7) - y_pred;

# Soma dos resíduos
sum_residuals = sum(residuals);

# Cálculo do erro padrão
std_error = std(residuals);

# Cálculo do coeficiente de determinação R^2
SStot = sum((df0(:, 7) - mean(df0(:, 7))).^2); % Soma total dos quadrados
SSres = sum(residuals.^2); % Soma dos quadrados dos resíduos
r_squared = 1 - (SSres / SStot); % Cálculo do R^2

# Exibir os resultados
fprintf('Soma dos Resíduos: %.4f\n', sum_residuals);
fprintf('Erro Padrão: %.4f\n', std_error);
fprintf('Coeficiente de Determinação R^2: %.4f\n', r_squared);

# Plotar a curva exponencial junto aos pontos de dados
figure;
scatter(x, df0(:, 7), 'r', 'o');
hold on;
plot(x_dense, y_pred_dense, 'b-', 'LineWidth', 2);
xlabel('Cost Bottle Water (pounds)');
ylabel('Happiness Levels');
title('Curva de Regressão Exponencial versus Happiness Levels (Cidades Menos Felizes)');
legend('Dados Originais', 'Ajuste Exponencial');
grid on;
hold off;




######  2.3

n = n_rows
for ip = 1:10
    AIC = n*log(SSR_/n) + 2 *(ip+1) *log(n)
​endfor















#### Exercício 3.1

xi = data_normalizado(:,7); %happiness
yi_Bottle = data_normalizado(:, 2); %cost bottle of water

figure(6);
hold on
grid on
plot(xi, yi_Bottle, 'o');
xlabel('happiness score');
ylabel('cost bottle of water');

# MMQ
n = length(xi);

a1 = (n*sum(xi.*yi_Bottle) - sum(xi)*sum(yi_Bottle))/(n*sum(xi.^2)-(sum(xi)^2));
a0 = mean(yi_Bottle) - a1*mean(xi);

%regressão linear
y_fitBottle = a0 + a1*xi;


plot(xi, y_fitBottle, 'r-')

%cálculo de métricas de qualidade de ajuste
residuals = yi_Bottle - y_fitBottle;
St = sum((yi_Bottle - mean(yi_Bottle)).^2)
Sr = sum((yi_Bottle-(y_fitBottle)).^2)
r2_3_1_Mod1 = (St-Sr)/St
rmse = sqrt(mean(residuals.^2)) %erro médio
s_yx = sqrt(Sr/(n-2))
s_y = sqrt(St/(n-1))

fprintf('y = %d + %d * x   ', a0, a1)


%modelo para expectativa de vida
figure(7);
hold on
grid on

xi_hap = data_normalizado(:,7); %happines
yi_life = data_normalizado(:, 4); %life expectancy

plot(xi_hap, yi_life, 'o');

xlabel('happiness score');
ylabel('life expectancy');

# MMQ
n = length(xi_hap);


a1 = (n*sum(xi_hap.*yi_life) - sum(xi_hap)*sum(yi_life))/(n*sum(xi_hap.^2)-(sum(xi_hap)^2));
a0 = mean(yi_life) - a1*mean(xi_hap);

hold on
plot(xi_hap, a1*xi_hap+a0, 'r')

%regressão linear
y_fit = a0 + a1*xi_hap


%cálculo de métricas de qualidade de ajuste
residuals = yi_life - y_fit;
St = sum((yi_life - mean(yi_life)).^2)
Sr = sum((yi_life-(y_fit)).^2)
r2_3_1_Mod2 = (St-Sr)/St
rmse = sqrt(mean(residuals.^2)) %erro médio
s_yx = sqrt(Sr/(n-2))
s_y = sqrt(St/(n-1))


fprintf('y = %d + %d * x   ', a0, a1)

# Resp: R2 do modelo 1 é melhor (0.3846 > 0.2046)


% Exercicio 3.2


# Preparar a matriz X com uma coluna de 1's para o intercepto
Water_Price = data_normalizado(:,2);  % Preço da água (variável independente 1)
Life_Expectancy = data_normalizado(:,4);  % Expectativa de vida (variável independente 2)
Happiness_Score = data_normalizado(:,7);  % Happiness Score (variável dependente)

X = [ones(length(Water_Price), 1), Water_Price, Life_Expectancy];

# Variável dependente (y)
y = Happiness_Score;


# Calcular os coeficientes de regressão (b) usando Mínimos Quadrados
# Função para resolver sistemas lineares com decomposição LU
function x = solveLU(L, U, b)
    # Substituição para frente: resolver L * z = b
    z = zeros(size(b));
    for i = 1:length(b)
        z(i) = (b(i) - L(i, 1:i-1) * z(1:i-1)) / L(i, i);
    endfor

    # Substituição para trás: resolver U * x = z
    x = zeros(size(b));
    for i = length(b):-1:1
        x(i) = (z(i) - U(i, i+1:end) * x(i+1:end)) / U(i, i);
    endfor
endfunction

# Preparação dos dados para o sistema linear
A = X' * X;  % Matriz do sistema
b = X' * y;  % Vetor do lado direito

# Decomposição LU
[L, U] = lu(A);

# Calcular os coeficientes b usando a função solveLU
coeficientes = solveLU(L, U, b);

# Exibir os coeficientes
disp('Coeficientes:');
disp(coeficientes);


# Exibir os coeficientes
disp('Coeficientes da Regressão Linear Múltipla:');
disp(b);

# Predizer os valores de Happiness Score usando o modelo ajustado
y_pred = X * coeficientes;

# Plotar os resultados
figure(32);
scatter3(Water_Price, Life_Expectancy, Happiness_Score,'b', 'filled');  % Scatter 3D dos dados reais
hold on;
plot3(Water_Price, Life_Expectancy, y_pred, 'ro');  % Linha de previsão
xlabel('Preço da Água');
ylabel('Expectativa de Vida');
zlabel('Happiness Score');
title('Regressão Linear Múltipla');
legend('Dados Reais', 'Modelo Ajustado');
hold off;


# Gerar o grid para o plano
[water_grid, life_grid] = meshgrid(linspace(min(Water_Price), max(Water_Price), 20), ...
                                   linspace(min(Life_Expectancy), max(Life_Expectancy), 20));

# Calcular os valores previstos para o plano
z_grid = coeficientes(1) + coeficientes(2) * water_grid + coeficientes(3) * life_grid;

# Plotar os dados e o plano de regressão
figure(33);
scatter3(Water_Price, Life_Expectancy, Happiness_Score, 'filled', 'b');  % Scatter dos dados reais
hold on;
surf(water_grid, life_grid, z_grid, 'FaceAlpha', 0.5);  % Plano de regressão com transparência

# Ajustar o gráfico
xlabel('Preço da Água');
ylabel('Expectativa de Vida');
zlabel('Happiness Score');
title('Plano de Regressão Linear Múltipla');
legend('Dados Reais', 'Plano de Regressão');
hold off;




%Calcular R² (Coeficiente de Determinação)
SST_reg_mul = sum((Happiness_Score - mean(Happiness_Score)).^2);  # Soma total dos quadrados
SSR_reg_mul = sum((Happiness_Score - y_pred).^2);  # Soma dos quadrados dos resíduos
R2_reg_mul = 1 - (SSR_reg_mul / SST_reg_mul);

%Calcular RMSE (Root Mean Squared Error)
RMSE_reg_mul = sqrt(mean((Happiness_Score - y_pred).^2));

%Calcular MAE (Mean Absolute Error)
MAE_reg_mul = mean(abs(Happiness_Score - y_pred));

# Exibir as métricas de ajuste
disp('Métricas de Ajuste:');
disp(['R²: ', num2str(R2_reg_mul)]);
disp(['RMSE: ', num2str(RMSE_reg_mul)]);
disp(['MAE: ', num2str(MAE_reg_mul)]);
disp(['Soma Total (SST): ', num2str(SST_reg_mul)]);
disp(['Soma Residual (SSR): ', num2str(SSR_reg_mul)])


# Resp: R2 da regressão multipla é melhor, logo esse modelo é melhor (0.39302 > 0.3846 > 0.2046)
