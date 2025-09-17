# IM-IFES
Gera um Excel consolidado a partir de:
- questionario.xlsx (estrutura: ID | Pergunta | Dimensão)
- respostas.xlsx (respostas brutas do Google Forms)

Regras implementadas:
- Fatores: 4->1.00, 3->0.75, 2->0.50, 1->0.25, 0->0.00, NA->ignorado
- Peso por pergunta válida (não-NA) dentro de uma dimensão e por resposta: Peso = 1/n_válidas
- Pontuação da pergunta = Fator * Peso
- Pontuação da Dimensão na unidade: soma das perguntas (por RespostaUID), depois média entre respostas da mesma unidade
- Pontuação da Dimensão na IFES: média das unidades da IFES para aquela dimensão
- IM da IFES: média aritmética das dimensões da IFES (todas com mesmo peso)
- Classificação por faixas:
    - 0.00–0.19 Incipiente
    - 0.20–0.39 Em planejamento
    - 0.40–0.59 Em implantação
    - 0.60–0.79 Implementado, porém otimizável
    - 0.80–1.00 Excelência

Recursos:
- Aceita arquivos .xlsx/.xls/.csv/.txt (questionário e respostas).
- Calcula IM por dimensão na unidade, por dimensão na IFES e IM da IFES (média do IM das dimensões).
- Exporta "dados agregados" para Excel.
- Gera relatórios TXT por IFES e por ifes_unidade contendo:
  (IFES)   A) Estatísticas por dimensão (Min, Q1, Mediana, Q3, Máximo, Desv.Pad., Média, IQR)
           B) Unidades com IM < mediana da IFES
           C) "Unidades Referência": IM >= Q3 da IFES
           D) Perguntas com mediana < mediana das respostas da dimensão (na IFES)
           E) "Perguntas Referência": mediana >= Q3 das respostas da dimensão (na IFES)
  (UNID)   A) Dimensões com IM da unidade < mediana(IFES, dimensão)
           B) Dimensões com IM da unidade < IM da dimensão na IFES (média)
           C) Perguntas da unidade com mediana < mediana das respostas da dimensão (na IFES)
           D) "Perguntas Referência" da unidade: mediana >= Q3 das respostas da dimensão (na IFES)
Observações nos TXT:
- "Abaixo da mediana" usa comparação estrita: < mediana
- "Acima do Q3" inclui empates: >= Q3
- Desvio-padrão amostral (ddof=1)
- No início de CADA TXT é exibido o IM (média do IM das dimensões).

Saída:
- Excel com dados agregados

Uso (exemplos)
    python im_calc.py \
        --questionario /caminho/questionario.xlsx \
        --respostas /caminho/respostas.xlsx \
        --saida /caminho/im_resultado.xlsx \
        --ifes-col "Informe a sua Instituição Federal de Ensino Superior" \
        --unidade-col "Informe unidade que você está representando"
