# Nano degree em Machine Learning
# Avaliação e validação
## Project: Projeto Final

### Instalação

Este projeto requer o **Python** e as seguintes bibliotecas instaladas:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- collections
- time

Você também precisa ter o seguinte software para executar o arquivo ipynb: [Jupyter Notebook](http://ipython.org/notebook.html)

Se você ainda não tem o Python instalado, é altamente recomendado que você instale a Distribuição Python chamada [Anaconda](http://continuum.io/downloads), que já possui todos os pacotes necessários instalados e alguns outros mais.

### Código


O arquivo com a implementação do projeto encontra-se  em `Projeto Ecocardiograma.ipynb`.
Para que o projeto execute de maneira correta, também é utilizado o arquivo `echo.csv`, que foi utilizado como o dataset no qual todo o projeto foi baseado e elaborado. Sem este arquivo, o notebook citado acima não funcionará.

### Run

Para executar, abra uma janela de terminal, e estando dentro do mesmo diretório onde os arquivos se encontram, digite um dos seguintes comandos: 

```bash
ipython notebook "Projeto Ecocardiograma.ipynb"
```  
ou 

```bash
jupyter notebook "Projeto Ecocardiograma.ipynb"
```

Esse comando irá abrir o notebook do projeto no seu navegador.

### Data

O Dataset utilizado constitui-se em um arqivo .dat (em formato csv) com 132 instâncias de pacientes. O dataset que utilizamos apenas diverge do original no ponto em que foi inserido uma linha inicial, onde descreve cada uma das colunas.

O dataset original encontra-se em [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/echocardiogram).

