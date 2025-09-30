O dataset do Titanic é um dos mais usados para aprendizado de machine learning porque é simples, 
mas rico em variáveis que permitem várias análises de classificação. 

1. Descrição detalhada dos dados
O dataset original vem do Kaggle - Titanic: Machine Learning from Disaster. 

Os principais arquivos são:
train.csv → contém os dados de treino, com a variável alvo (Survived).
test.csv → contém dados sem a variável alvo (para prever).


Variável alvo
Survived: se o passageiro sobreviveu (1) ou não (0). É uma variável binária.

Principais variáveis preditoras

PassengerId → Identificador único do passageiro (não é útil para previsão).
Pclass → Classe do bilhete (1ª, 2ª ou 3ª). Representa nível socioeconômico:
    1 = alta classe,
    2 = média,
    3 = baixa.

Name → Nome completo. Pode ser processado para extrair títulos como "Mr.", "Mrs.", "Miss." (que ajudam a inferir idade ou status social).

Sex → Sexo (male/female).

Age → Idade em anos (muitos valores ausentes).

SibSp → Quantos irmãos/cônjuges a bordo.

Parch → Quantos pais/filhos a bordo.

Ticket → Número do bilhete (pouco informativo bruto, mas pode ser categorizado).

Fare → Tarifa paga pela passagem.

Cabin → Número da cabine (muitos ausentes, mas pode indicar localização no navio).

Embarked → Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton).

2. Problema de negócio

O naufrágio do Titanic, em 1912, resultou em mais de 1.500 mortes. 
O objetivo do problema é prever a probabilidade de sobrevivência de cada passageiro a partir de suas características.


Objetivos de negócio

Modelo preditivo: Criar um modelo de classificação binária que estime se um passageiro sobreviveu (1) ou não (0).

Compreensão de fatores críticos: Identificar quais variáveis influenciam a sobrevivência 
(ex.: mulheres e crianças tiveram prioridade, passageiros da 1ª classe tiveram mais chance).


Características do problema

Tarefa de ML: Classificação supervisionada binária.

Desafios:

Dados ausentes (especialmente em Age e Cabin).
Variáveis categóricas que precisam de transformação (Sex, Embarked).
Desbalanceamento moderado (cerca de 38% sobreviveram, 62% não).
Algumas variáveis pouco informativas (Ticket, PassengerId).

Métricas comuns de avaliação

Accuracy → usado no Kaggle.
Precision, Recall e F1-score → úteis para balancear erros de classificação.
ROC-AUC → medir qualidade geral da separação entre sobreviventes e não sobreviventes.
