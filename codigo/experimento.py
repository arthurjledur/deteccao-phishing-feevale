"""Avaliação 2 - Aprendizado de Máquina | Arthur Juchem Ledur.
Execução: python codigo/experimento.py (a partir da pasta do projeto).
Exige acesso à internet somente para baixar o conjunto original da UCI na primeira execução.
"""
from pathlib import Path
import io, zipfile, json, time, sys, platform, hashlib
import argparse
import os
from datetime import date
from urllib.request import urlopen
from scipy import __version__ as scipy_version
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import sklearn, joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'dados/original'; PROC=ROOT/'dados/processados'; RES=ROOT/'resultados'; FIG=ROOT/'graficos'; MOD=ROOT/'modelos'
for d in (RAW,PROC,RES,FIG,MOD):d.mkdir(parents=True,exist_ok=True)
URL='https://archive.ics.uci.edu/static/public/327/phishing+websites.zip'

def obter_arff():
    name = 'Training Dataset.arff'
    output = RAW / name
    if output.is_file():
        return output
    print('Baixando arquivo original da UCI...')
    with urlopen(URL, timeout=90) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        matches = [item for item in source.namelist() if Path(item).name == name]
        if len(matches) != 1:
            raise ValueError('O ZIP oficial deve conter exatamente um Training Dataset.arff.')
        output.write_bytes(source.read(matches[0]))
    return output


def carrega():
    file=obter_arff()
    array,metadata=arff.loadarff(file)
    df=pd.DataFrame(array)
    for c in df:
        if df[c].dtype=='object':
            df[c]=df[c].map(lambda v:v.decode('utf-8') if isinstance(v,bytes) else v)
        df[c]=pd.to_numeric(df[c],errors='raise')
    if df.shape != (11055, 31):
        raise ValueError(f'Dimensões inesperadas no ARFF oficial: {df.shape}')
    target=next((c for c in df if c.casefold()=='result'),None)
    if target is None:raise ValueError('Atributo target Result não encontrado no conjunto original')
    # A UCI codifica: -1 = phishing, 1 = legítimo.
    if set(df[target].dropna().unique())!={-1,1}:raise ValueError(f'Classes inesperadas: {df[target].unique()}')
    if df.isna().any().any():raise ValueError('Foram detectadas ausências. Investigar antes de imputar.')
    df.to_csv(PROC/'phishing_tratado.csv',index=False)
    return df,target,file

def barplot(series,title,path,xlabel='',rotation=0):
    fig,ax=plt.subplots(figsize=(8,4.4))
    bars=ax.bar([str(x) for x in series.index],series.values)
    ax.bar_label(bars,padding=3)
    ax.set_title(title);ax.set_ylabel('Quantidade');ax.set_xlabel(xlabel)
    ax.tick_params(axis='x',labelrotation=rotation)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Executa a avaliação supervisionada.')
    parser.add_argument('--github', default='', help='URL do repositório da entrega')
    args = parser.parse_args()
    df,target,rawfile=carrega()
    X=df.drop(columns=target);y=df[target].astype(int)
    table=[]
    for col in df.columns:
        s=df[col];freq=s.value_counts(dropna=False)
        table.append({'atributo':col,'papel':'target' if col==target else 'entrada',
                      'tipo':'categórico codificado como inteiro','valores':', '.join(map(str,sorted(s.dropna().unique().astype(int)))),
                      'faltantes':int(s.isna().sum()),'qtd_valores_distintos':int(s.nunique()),
                      'categoria_majoritaria_pct':float(100*freq.max()/len(df))})
    perf=pd.DataFrame(table)
    perf.to_csv(RES/'caracterizacao_atributos.csv',index=False)
    # 'Mais desbalanceado' = maior participação de um único valor entre atributos de entrada.
    most=perf.query("papel=='entrada'").sort_values(['categoria_majoritaria_pct','atributo'],ascending=[False,True]).iloc[0]
    counts=y.value_counts().sort_index();barplot(counts.rename(index={-1:'Phishing (-1)',1:'Legítimo (1)'}),
                     'Distribuição das classes',FIG/'classes.png')
    top=perf.query("papel=='entrada'").sort_values('categoria_majoritaria_pct',ascending=False).head(12)
    fig,ax=plt.subplots(figsize=(11,5));ax.barh(top['atributo'][::-1],top['categoria_majoritaria_pct'][::-1]);ax.set_xlim(0,100)
    ax.set_xlabel('Frequência da categoria majoritária (%)');ax.set_title('Atributos com maior concentração de valores')
    fig.tight_layout();fig.savefig(FIG/'desbalanceamento.png',dpi=160);plt.close(fig)
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    (PROC/'indices_divisao.json').write_text(json.dumps({'train':list(map(int,Xtrain.index)),'test':list(map(int,Xtest.index))},ensure_ascii=False),encoding='utf-8')
    algos={
        'Arvore de Decisao':DecisionTreeClassifier(random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=1),
        'Regressao Logistica':Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(max_iter=3000,random_state=42))])
    }
    # 5 folds SÓ no treinamento; teste reservado para avaliação final independente.
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    linhas=[];resumos=[];per_class=[];predictions=[]
    for label,model in algos.items():
        print('Executando:',label,flush=True)
        for fold,(itr,ival) in enumerate(folds.split(Xtrain,ytrain),start=1):
            xx,yy=Xtrain.iloc[itr],ytrain.iloc[itr]
            t0=time.perf_counter();model.fit(xx,yy);elapsed=time.perf_counter()-t0
            yp=model.predict(Xtrain.iloc[ival]);yt=ytrain.iloc[ival]
            p,r,f,_=precision_recall_fscore_support(yt,yp,average='macro',zero_division=0)
            linhas.append({'algoritmo':label,'fold':fold,'acuracia':accuracy_score(yt,yp),
                           'precisao_macro':p,'recall_macro':r,'f1_macro':f,'treinamento_segundos':elapsed})
        part=pd.DataFrame([v for v in linhas if v['algoritmo']==label]);row={'algoritmo':label}
        for name in ('acuracia','precisao_macro','recall_macro','f1_macro','treinamento_segundos'):
            row[name]=float(part[name].mean());row[name+'_desvio']=float(part[name].std(ddof=1))
        resumos.append(row)
        t0=time.perf_counter();model.fit(Xtrain,ytrain);tf=time.perf_counter()-t0
        yp=model.predict(Xtest)
        report=classification_report(ytest,yp,labels=[-1,1],output_dict=True,zero_division=0)
        for code,human in [(-1,'Phishing'),(1,'Legítimo')]:
            record=report[str(code)]
            per_class.append({'algoritmo':label,'classe':human,'codigo':code,
                              'precisao':record['precision'],'recall':record['recall'],
                              'f1':record['f1-score'],'suporte':int(record['support'])})
        for index,real,pred in zip(Xtest.index,ytest,yp):
            predictions.append({'algoritmo':label,'indice_original':int(index),'real':int(real),'predito':int(pred)})
        matrix=confusion_matrix(ytest,yp,labels=[-1,1]);fig,ax=plt.subplots(figsize=(5.2,4.5))
        im=ax.imshow(matrix,cmap='Blues');fig.colorbar(im,ax=ax)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',fontsize=12)
        ax.set_xticks([0,1],labels=['Phishing','Legítimo']);ax.set_yticks([0,1],labels=['Phishing','Legítimo'])
        ax.set_xlabel('Predito');ax.set_ylabel('Real');ax.set_title(label);fig.tight_layout()
        fig.savefig(FIG/('confusao_'+label.lower().replace(' ','_')+'.png'),dpi=160);plt.close(fig)
        slug=label.lower().replace(' ','_');joblib.dump(model, MOD / (slug + '.joblib'), compress=3)
        row['teste_acuracia']=float(accuracy_score(ytest,yp));row['treino_final_segundos']=tf
    pd.DataFrame(linhas).to_csv(RES/'metricas_por_fold.csv',index=False)
    ranking=pd.DataFrame(resumos).sort_values(['f1_macro','acuracia'],ascending=False)
    ranking.to_csv(RES/'comparacao_algoritmos.csv',index=False)
    pd.DataFrame(per_class).to_csv(RES/'metricas_por_classe_teste.csv',index=False)
    pd.DataFrame(predictions).to_csv(RES/'predicoes_teste.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4.7));metrics=['acuracia','precisao_macro','recall_macro','f1_macro']
    x=np.arange(len(ranking));w=.2
    for i,m in enumerate(metrics):ax.bar(x+(i-1.5)*w,ranking[m],width=w,label=m.replace('_macro','').title())
    ax.set_xticks(x,ranking['algoritmo']);ax.set_ylim(0,1.08);ax.legend(ncol=4,loc='upper center');ax.set_title('Comparação média - validação cruzada (5 folds)')
    fig.tight_layout();fig.savefig(FIG/'comparativo_modelos.png',dpi=160);plt.close(fig)
    frequencias = df[most['atributo']].value_counts()
    vetores_treino = set(Xtrain.itertuples(index=False, name=None))
    repetidos_teste = sum(v in vetores_treino for v in Xtest.itertuples(index=False, name=None))
    info={'github':args.github, 'data':date.today().strftime('%d/%m/%Y'),
          'vetores_teste_repetidos_no_treino': repetidos_teste,
          'categoria_mais_concentrada': int(frequencias.index[0]),
          'contagem_mais_concentrada': int(frequencias.iloc[0]),
          'origem':'UCI Phishing Websites, ID 327','url':'https://archive.ics.uci.edu/dataset/327/phishing+websites',
          'instancias':len(df),'atributos_entrada':X.shape[1],'atributos_totais':df.shape[1],
          'target':target,'classes':{'-1':'Phishing','1':'Legítimo'},'distribuicao':{str(k):int(v) for k,v in counts.items()},
          'atributos_com_faltantes':int((df.isna().sum()>0).sum()),'total_faltantes':int(df.isna().sum().sum()),
          'mais_desbalanceado':{'nome':str(most['atributo']),'frequencia_majoritaria_pct':float(most['categoria_majoritaria_pct'])},
          'divisao':{'treino':len(Xtrain),'teste':len(Xtest),'estratificada':True,'semente':42},
          'modelo_melhor_validacao':str(ranking.iloc[0]['algoritmo']),'modelo_pior_validacao':str(ranking.iloc[-1]['algoritmo']),
          'arquivo_original_sha256':hashlib.sha256(rawfile.read_bytes()).hexdigest(),
          'ambiente':{'python':sys.version.split()[0],'sistema':platform.platform(),'sklearn':sklearn.__version__,'pandas':pd.__version__,
                      'numpy':np.__version__, 'scipy':scipy_version,
                      'matplotlib':matplotlib.__version__, 'joblib':joblib.__version__,
                      'cpu':os.getenv('PHISHING_CPU') or platform.processor() or 'indisponível',
                      'cpus_logicas':os.cpu_count(), 'ram_gib':os.getenv('PHISHING_RAM_GIB', 'indisponível')}}
    (RES/'resumo_execucao.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    gerar_relatorio(info,ranking,pd.DataFrame(per_class),perf)
    print('\nEXPERIMENTO CONCLUIDO\n',ranking[['algoritmo']+metrics+['treinamento_segundos']].to_string(index=False))

def gerar_relatorio(info, ranking, per_class, atributos):
    # PDF autogerado a partir dos resultados reais: nunca contém números fictícios.
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import cm
    from xml.sax.saxutils import escape
    pdf=ROOT/'Relatorio_Final_Resultados_Reais.pdf';doc=SimpleDocTemplate(str(pdf),pagesize=(595,842),rightMargin=43,leftMargin=43,topMargin=44,bottomMargin=44)
    sty=getSampleStyleSheet();sty.add(ParagraphStyle(name='smallx',parent=sty['Normal'],fontSize=8.2,leading=10,spaceAfter=0));sty['BodyText'].leading=14
    E=[]
    def p(t,style='BodyText'):E.append(Paragraph(t,sty[style]));E.append(Spacer(1,6))
    def h(t):p(escape(t),'Heading2')
    def table(rows,widths=None):
        ts=Table([[Paragraph(escape(str(v)),sty['smallx']) for v in r] for r in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        ts.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E4ECF6')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#CED5DD')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#FAFBFC')]),('LEFTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));E.append(ts);E.append(Spacer(1,12))
    p('DETECÇÃO DE PHISHING COM APRENDIZADO DE MÁQUINA','Title')
    p(f"Arthur Juchem Ledur | Universidade Feevale | Avaliação 2 de Aprendizado de Máquina | {info['data']}",'Normal')
    h('1. Objetivo e dataset')
    p('Comparar três classificadores supervisionados para identificar sites legítimos e de phishing a partir de 30 atributos categóricos previamente extraídos. Fonte: UCI Machine Learning Repository, dataset 327 (Phishing Websites). URL: '+info['url'])
    p('Atributo classe (target): Result. Codificação verificada no dataset: -1 = phishing; 1 = legítimo. Os valores de entrada -1, 0 e 1 representam categorias das características, não medidas contínuas.')
    h('2. Caracterização dos dados')
    p(f"Total: {info['instancias']} instâncias, {info['atributos_entrada']} atributos de entrada e uma classe (total {info['atributos_totais']} colunas). Valores faltantes: {info['total_faltantes']}; atributos com faltantes: {info['atributos_com_faltantes']}.")
    p('Distribuição: '+', '.join(f'{k}: {v}' for k,v in info['distribuicao'].items())+'. O atributo de entrada mais concentrado em um único valor é '+info['mais_desbalanceado']['nome']+f" ({info['mais_desbalanceado']['frequencia_majoritaria_pct']:.2f}% de frequência da categoria majoritária). O valor majoritário é {info['categoria_mais_concentrada']}, presente em {info['contagem_mais_concentrada']} registros.")
    E.append(Image(str(FIG/'classes.png'),width=12.7*cm,height=7*cm))
    E.append(Image(str(FIG/'desbalanceamento.png'),width=14.5*cm,height=6.6*cm))
    E.append(PageBreak())
    h('3. Tabela de atributos (valores observados)')
    rows=[['Atributo','Tipo','Valores','Ausentes']]
    for _,r in atributos.iterrows():rows.append([r['atributo'],'Categórico inteiro',r['valores'],str(r['faltantes'])])
    table(rows,[5.4*cm,4.3*cm,4.1*cm,2.2*cm])
    E.append(PageBreak())
    h('4. Pré-processamento e desenho do experimento')
    p('Dados originais em ARFF convertidos para CSV mantendo integralmente as linhas, colunas e códigos de classe. Tipos nominais convertidos a inteiros, sem criação de colunas derivadas. Não houve imputação por ausência de valores faltantes. Para Regressão Logística, padronização StandardScaler aplicada somente dentro do pipeline de treinamento. Para as árvores, atributos não padronizados. Dados separados estratificadamente em 80% de treino e 20% de teste, semente 42. Validação cruzada estratificada em 5 partes somente no treinamento; métricas são médias dos 5 folds, sem acesso ao teste. Tempos incluem somente ajuste (fit) em cada fold, medidos com time.perf_counter em segundos.')
    p('Precisão, recall e F1 na tabela são médias macro (média simples das classes); acurácia corresponde à fração de exemplos classificados corretamente. Os modelos são comparados pelas mesmas divisões, sem ajuste de hiperparâmetros no conjunto de teste.')
    h('5. Resultados reais - médias da validação cruzada')
    table([['Algoritmo','Acurácia','Precisão','Recall','F1','Treino (s)']]+[[r.algoritmo]+[f'{getattr(r,k):.4f}' for k in ('acuracia','precisao_macro','recall_macro','f1_macro')]+[f'{r.treinamento_segundos:.4f}'] for r in ranking.itertuples(index=False)], [4.5*cm,2.2*cm,2.2*cm,2.0*cm,1.8*cm,2.3*cm])
    E.append(Image(str(FIG/'comparativo_modelos.png'),width=15.9*cm,height=8.2*cm))
    E.append(PageBreak())
    h('6. Análise dos modelos e classes no conjunto de teste')
    best=ranking.iloc[0];worst=ranking.iloc[-1]
    p(f"Maior F1 macro média: {best['algoritmo']}, F1={best['f1_macro']:.4f}, acurácia={best['acuracia']:.4f}, precisão={best['precisao_macro']:.4f}, recall={best['recall_macro']:.4f}, treino médio={best['treinamento_segundos']:.4f}s. Menor F1 macro média: {worst['algoritmo']}, F1={worst['f1_macro']:.4f}, acurácia={worst['acuracia']:.4f}, precisão={worst['precisao_macro']:.4f}, recall={worst['recall_macro']:.4f}, treino médio={worst['treinamento_segundos']:.4f}s.")
    table([['Algoritmo','Classe','Precisão','Recall','F1','Suporte']]+[[r.algoritmo,r.classe,f'{r.precisao:.4f}',f'{r.recall:.4f}',f'{r.f1:.4f}',str(r.suporte)] for r in per_class.itertuples(index=False)], [4.7*cm,2.7*cm,2.2*cm,2.2*cm,2*cm,2.2*cm])
    good=per_class.sort_values(['f1','recall'],ascending=False).iloc[0];bad=per_class.sort_values(['f1','recall']).iloc[0]
    p(f"Entre as combinações algoritmo/classe avaliadas, o melhor F1 foi para {good['classe']} com {good['algoritmo']} ({good['f1']:.4f}); o pior foi para {bad['classe']} com {bad['algoritmo']} ({bad['f1']:.4f}). A taxa de falsos negativos para phishing deve ser considerada especialmente relevante no uso prático.")
    for model in (best['algoritmo'], worst['algoritmo']):
        classes = per_class[per_class['algoritmo'] == model].sort_values('f1')
        low, high = classes.iloc[0], classes.iloc[-1]
        p(f"{model}: maior F1 para {high['classe']} ({high['f1']:.4f}) e menor para {low['classe']} ({low['f1']:.4f}).")
    best_slug = best['algoritmo'].lower().replace(' ', '_')
    E.append(Image(str(FIG / f'confusao_{best_slug}.png'), width=9*cm, height=7.8*cm))
    E.append(PageBreak())
    h('7. Discussão e limitações')
    p(f"No conjunto reservado, {best['algoritmo']} atingiu {best['teste_acuracia']:.2%} de acurácia. "
      'O artigo indicado pela UCI, de Mohammad, Thabtah e McCluskey (2012), investigou regras para '
      'extrair indicadores das páginas e avaliou sua relevância para detectar phishing. Neste trabalho, '
      'esses indicadores já vêm codificados e servem de entrada a três classificadores. O artigo trata '
      'da extração e relevância das características, enquanto a comparação aqui mede desempenho preditivo.')
    p(f"O particionamento por linha foi mantido conforme o projeto original. "
      f"{info['vetores_teste_repetidos_no_treino']} dos {info['divisao']['teste']} registros de teste "
      'têm um vetor de entrada idêntico a algum vetor de treino. Essa sobreposição pode elevar as '
      'métricas. A base é histórica e não contém URLs brutas, o que limita a generalização para novas campanhas.')
    h('8. Reprodutibilidade e entrega')
    ambiente=info['ambiente']
    p(escape(f"Ambiente: {ambiente['sistema']}, {ambiente['cpu']}, "
             f"{ambiente['cpus_logicas']} processadores lógicos, RAM (GiB): {ambiente['ram_gib']}. "
             f"Python {ambiente['python']}, scikit-learn {ambiente['sklearn']}, pandas {ambiente['pandas']}, "
             f"NumPy {ambiente['numpy']}, SciPy {ambiente['scipy']}, "
             f"Matplotlib {ambiente['matplotlib']}, joblib {ambiente['joblib']}."))
    p('O código, dados, modelos, figuras e logs acompanham o PDF. O README reúne os comandos e '
      'o arquivo resultados/resumo_execucao.json registra a divisão e o SHA-256 do ARFF original.')
    if info['github']:
        url = escape(info['github'], {'"': '&quot;'})
        p(f'GitHub: <a href="{url}" color="#1A5593">{url}</a>')
    else:
        p('GitHub: endereço pendente de confirmação.')
    h('Referências')
    p('Mohammad, R. e McCluskey, L. (2012). Phishing Websites. UCI Machine Learning Repository. '
      '<a href="https://doi.org/10.24432/C51W2X" color="#1A5593">DOI 10.24432/C51W2X</a>. CC BY 4.0.')
    p('Mohammad, R. M., Thabtah, F. e McCluskey, L. (2012). An Assessment of Features Related to '
      'Phishing Websites Using an Automated Technique. ICITST, p. 492-497. '
      '<a href="https://ieeexplore.ieee.org/document/6470857" color="#1A5593">IEEE 6470857</a>.')
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#41556B'))
        canvas.drawString(43, 27, 'Universidade Feevale | Aprendizado de Máquina')
        canvas.drawRightString(552, 27, str(document.page))
    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    print('Relatório PDF gerado:',pdf)

if __name__=='__main__':main()
