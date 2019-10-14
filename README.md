## ML-Model-Flask-Deployment
Esse arquivo foi criado com intuito de servir como avaliação no processo seletivo da empresa SciCrop

### Pre-Requisitos
Deve ter instalado Scikit Learn, Pandas, pickle (Para ser usado na primeira parte do processo, onde geramos os modelos de machine learning) e Flask (Para API) instalados.

### Estrutura do projeto
Projeto foi dividido em duas partes.
1. Pasta Modelo Machine-Learning, onde temos os codigos usados para gerar e salvar(com pickle) nosso modelo de ML.

2. Aplicação Web com:
    1. app.py - Esse arquivo contem nossa aplicação FLASK. Ele é o responsável por fazer a integração entre a API e a pagina HTML.
    2. pasta /static/modelos/ contem os modelos serializados de machine learning.
    3. templates - Essa pasta contem o site principal da aplicação.



### Rodando o projeto
#### Modelo de Machine Learning
1. Apos instalado as dependências, precisamos rodar o código com o notebook jupyter ou através do comando 'python projecoes_estatisticas.py' com os arquivos (apy.csv e API_NY.GDP.MKTP.CD_DS2_en_csv_v2_247793.csv) na pasta.
2. Teremos como resultado
    1. Arquivo india_crop_gdp_1997_2015.csv com os dados das duas tabelas.
    2. Arquivo model_1_FindCrop.sav contendo a serialização do modelo que preve o cultivo.
    3. Arquivo model_2_FindProduction.sav que contem a serialização de um modelo que preve a produção agrícola de um cultivo.
    
#### Aplicação WEB

1. Abra o terminal na pasta da aplicação e digite o seguinte comando no terminal para iniciar a aplicação Flask API:
```
python app.py
```

Irá aparecer a URL que você deve acessar a aplicação; Por default, o flask roda na port 5000.

2. Navegue para URL http://localhost:5000

Deve aparecer essa tela:

![alt text](https://i.ibb.co/H2G9Sb3/aaaa.png)

Apos digitar os valores desejados e usar o botão para prever(no exemplo previ o cultivo) temos a seguinte tela:
![alt text](https://i.ibb.co/82fDhL2/bbbb.png)


