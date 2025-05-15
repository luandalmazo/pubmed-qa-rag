#!/bin/bash
#SBATCH --job-name=qa_pdf_pubmed
#SBATCH --output=logs/saida_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                
#SBATCH --mem=64G                 
#SBATCH --cpus-per-task=8          
#SBATCH --time=00:30:00        


python terminal.py
