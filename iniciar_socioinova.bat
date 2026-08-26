@echo off
title SocioInova RAG - Inicializacao Rapida
cls

echo ======================================================
echo    SocioInova RAG - Estacao de Pesquisa (UFBA)
echo    Pesquisador: Jonata Franca Bittencourt
echo ======================================================
echo.

echo [1/3] A verificar o Servidor Ollama...
:: Inicia o Ollama em segundo plano caso nao esteja aberto
start /min ollama serve

echo [2/3] A ativar o Ambiente Virtual Python...
:: Ajuste o caminho abaixo se a sua pasta for diferente
cd /d C:\Users\jonat\analise_ifs
call venv_tese\Scripts\activate

echo [3/3] A iniciar a Interface Multi-Modelo...
echo Aguarde... O navegador abrira automaticamente.
streamlit run app.py

pause