from . import engine
from django.shortcuts import render
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # tf-idf
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd



def index(request):
  context = {
    'heading' : 'Home'
  }
  # if request.method == 'POST':
  #   print(request.POST['kalimat']);
  #   context['kalimat'] = request.POST['kalimat']
  # else :
  #   print("method geet")
  return render(request, 'index.html', context)



def proses(request):
  if request.method == 'POST':
    kalimat = request.POST['kalimat']
    params_angka = request.POST['input_kalimat']
    pemisal_kalimat = engine.splitParagraphIntoSentences(kalimat)
    simpan_sementara_isi_berita = list()
    berita_asli = list()
    for per_kalimat in pemisal_kalimat:
        simpan_sementara_isi_berita.append(engine.preprocessing(per_kalimat.strip()))
        berita_asli.append(per_kalimat.strip())

    factory = StopWordRemoverFactory()
    stopwords = factory.get_stop_words()
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords,smooth_idf=False, norm=None)
    X = tfidf_vectorizer.fit_transform(simpan_sementara_isi_berita)
    return_TFIDF  = pd.DataFrame(X.toarray(), columns=tfidf_vectorizer.get_feature_names()).T

    return_LSA = pd.DataFrame(engine.LSA(X)).T
    return_CLSA = pd.DataFrame(engine.CLSA(X)).T

    rank_LSA = engine.sum_frame_by_column(return_LSA, 'total_score_document', [i for i in range(len(return_CLSA[0]))])
    rank_CLSA = engine.sum_frame_by_column(return_CLSA, 'total_score_document', [i for i in range(len(return_CLSA[0]))])

    docs = [str(x) for x in simpan_sementara_isi_berita]
    documentNames = list()
    for i,simpan_sementara_isi_berita in enumerate(docs):
        documentNames.append("Document_{}".format(i+1))
    
    return_LSA['documentNames'] = documentNames
    return_LSA['rank'] = return_LSA['total_score_document'].rank(method='first', ascending=False).astype(int)
    return_CLSA['documentNames'] = documentNames
    return_CLSA['rank'] = return_CLSA['total_score_document'].rank(method='first', ascending=False).astype(int)

    aftersort_LSA = rank_LSA.sort_values(['total_score_document'], ascending=[False])
    aftersort_LSA['rank'] = range(1, len(aftersort_LSA) + 1)
    aftersort_CLSA = rank_CLSA.sort_values(['total_score_document'], ascending=[False])
    aftersort_CLSA['rank'] = range(1, len(aftersort_CLSA) + 1)

    sentences_lsa = engine.summary_sentence(berita_asli, X, params_angka, types='lsa')
    sentences_clsa = engine.summary_sentence(berita_asli, X, params_angka, types='clsa')

    hasil = {
      'sebelum_preprocessing' : kalimat,
      'jumlah_masukan_kalimat' : params_angka,
      'sesudah_preprocessing' : docs,
      'tables_TFIDF' : [return_TFIDF.to_html()],
      'tables_LSA' : [return_LSA.to_html()] ,
      'tables_CLSA' : [return_CLSA.to_html()],
      'sum_tables_LSA' : [aftersort_LSA.to_html()],
      'sum_tables_CLSA' : [aftersort_CLSA.to_html()],
      'sentences_lsa' : sentences_lsa,
      'sentences_clsa' : sentences_clsa
    }
    return render(request, 'proses.html', hasil)
  else:
    return render(request, 'index.html')