# IM-IFES
Gera um Excel consolidado (4 abas) a partir de:
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

Saída:
- Excel com 5 abas: dados, consolidacao_dimensao_unidade, consolidacao_dimensao_ifes, consolidacao_ifes, dados agregados

Uso (exemplos)
    python im_calc.py \
        --questionario /caminho/questionario.xlsx \
        --respostas /caminho/respostas.xlsx \
        --saida /caminho/im_resultado.xlsx \
        --ifes-col "Informe a sua Instituição Federal de Ensino Superior" \
        --unidade-col "Informe unidade que você está representando"

Requer: pandas, numpy, xlsxwriter
