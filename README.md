# Detecção de phishing com aprendizado de máquina

Avaliação 2 de Aprendizado de Máquina · Universidade Feevale  
Arthur Juchem Ledur

Comparação de Árvore de Decisão, Random Forest e Regressão Logística na classificação supervisionada de sites. Os modelos recebem os 30 indicadores já extraídos pela UCI. O atributo classe é `Result`: `-1` indica phishing e `1` indica site legítimo.

## Reprodução

Use Python 3.12 e execute os comandos a partir da raiz. A opção `--github` inclui o endereço da entrega no PDF e é dispensável para uma reprodução local.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -u codigo/experimento.py --github https://github.com/arthurjledur/deteccao-phishing-feevale 2>&1 |
  Tee-Object resultados/execucao.log
```

A execução usa o ARFF incluído no pacote. Se o arquivo estiver ausente, o script busca o ZIP oficial e extrai o ARFF principal. A partição treino/teste 80/20 e os cinco folds da validação são estratificados e usam a semente 42. O teste reservado é avaliado depois do ajuste final. Os tempos de treino são medidos em segundos e variam com o ambiente.

## Arquivos

| Caminho | Conteúdo |
| --- | --- |
| `dados/original/Training Dataset.arff` | Dados oficiais no formato original |
| `dados/processados/` | CSV convertido e índices da divisão |
| `codigo/experimento.py` | Carregamento, pré-processamento, treinamento, avaliação e geração do PDF |
| `modelos/` | Três modelos ajustados, em formato Joblib |
| `resultados/` | Log, métricas por fold/algoritmo/classe, predições e configuração da execução |
| `graficos/` | Distribuições, métricas agregadas e matrizes de confusão |
| `Relatorio_Final_Resultados_Reais.pdf` | Relatório da Avaliação 2 |

A entrega do professor também exige o registro exclusivo do dataset na tabela da turma e o link do repositório publicado no PDF. O ZIP contém a árvore do repositório pronta para publicação.

## Origem dos dados

Mohammad, R. e McCluskey, L. (2012). *Phishing Websites*. UCI Machine Learning Repository. [DOI 10.24432/C51W2X](https://doi.org/10.24432/C51W2X). Licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). O CSV incluído é uma conversão do ARFF, mantendo as linhas, os atributos e os códigos originais.
