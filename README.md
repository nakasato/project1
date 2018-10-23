# Estimando a qualidade de vinho: regressão e classificação

## Problema:
_"O presente problema se refere aos dados de vinhos portugueses "Vinho Verde", que possuem variantes de vinho branco e tinto. Devido a questões de privacidade, apenas variáveis
físico-químicas (input) e sensoriais (output) estão disponíveis (por exemplo, não há dados sobre tipo de uva, marca do vinho, preço de venda, etc). Nosso objetivo é criar um modelo para estimar a qualidade do vinho."_


## Resolução:
**1. Análise exploratória:** está contida no arquivo `analise_exploratoria.ipynb`. Síntese dos passos:
- Abertura do dataset e checagem de tipos, colunas e registros nulos;
- Correção da variável _alcohol_;
- Estatística descritiva básica de todo dataframe
- Distribuição da variável dependente _quality_;
- Heatmap d e correlações entre todas as variáveis numéricas;
- Matriz de dispersão das variáveis numéricas colorida pela gradação de qualidade do vinho;
- Limpeza de outliers na _density_ (descobertos no passo anterior);
- Pairplot de todas as variáveis, corrigidas, distinguidas pelo _type_ do vinho.

**Nota:** os tratamentos de sanitização que se fizeram necessários nesta etapa serão replicados nas próximas, de modelagem propriamente.

**2.a. Estratégia de modelagem:** a estratégia de modelagem foi grandemente influenciada pela distribuição de _y = quality_ observada no quarto tópico da seção anterior. A variável tomava valores discretos de 3 a 9, de sorte que é intuitivo supor que uma solução à regressão da forma _quality = f(density, pH, alcohol,...)_ fosse uma boa alternativa. Apesar da variedade de modelos de regressão testados, as soluções deixaram a desejar. Assim, modificamos um pouco o problema, criando categorias baseadas em ranges da qualidade, e modelamos como um problema de classificação: primeiro considerando três tipos de qualidade diferentes, depois apenas dois (bom ou ruim). Portanto, em resumo:
- Resolução como regressão: `modelos_de_regressao.ipynb`;
- Resolução como classificação em 3 grupos: `modelos_de_classificacao_3grupos.ipynb`;
- Resolução como classificação em 2 grupos: `modelos_de_classificacao_2grupos.ipynb`.

**2.b. Função de custo:** funções de custo comuns baseadas erro quadrático médio (para regressão) e gradiente descendente (para classificação com redes neurais).

**2.c. Critério de seleção do modelo final:** dois critérios foram utilizados, um para cada tipo de problema:
- Regressão: menor RMSE entre um modelo selecionado após baterias de testes via _auto-sklearn_, _regressão linear à múltiplas variáveis_, _random forest_ e _decision trees_. A regressão linear se saiu melhor, com _RMSE = 0.64_ e _RMSE normalizado = 0.11_.
- Classificação: maior acurácia entre os dados de teste e os preditos. Os modelos testados foram o _auto-sklearn_ (busca entre vários modelos), _redes neurais_, _logistic regression_, _decision trees_. Para o problema de classificação em 3 grupos de qualidade, o melhor modelo foi o obtido pelo auto-sklearn, com _acurácia = 0.70_. No caso da classificação em 2 grupos, os modelos usados foram _logistic regression_, _redes neurais_ e _decision trees_, e os dois últimos empataram com _acurácia = 0.82_.
De modo geral, podemos concluir que converter esse problema para um problema de classificação dicotômica é o que gerou melhor resultado preditivo final.

**2.d. Critério de validação:** divisão randomizada do dataset em treino/teste, com 30% dos dados para teste.

**2.e. Evidências de qualidade:** no problema de regressão, na verdade, o RMSE ficou acima do esperado, com seu valor normalizado beirando 11% do range de y teste. Os erros, no entanto, mantiveram uma distribuição normal, não dentada ou viciada, mas bem distribuída em torno de zero em todos os resultados, o que é positivo.

O caso da classificação em 3 grupos original também deixou a desejar, com acurácias alcançando no máximo 70%. O melhor caso, o modelo de classificação em 2 grupos, teve um desempenho aceitável, com acurácia elevada e uma matriz de confusão bem ponderada com falsos-positivos/negativos.

Em todos as ocasiões, um melhor tuning de hiperparâmetros via GridSearch provavelmente melhoraria de forma significativa os resultados.
