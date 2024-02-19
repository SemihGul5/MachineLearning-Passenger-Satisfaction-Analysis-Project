# MachineLearning-Passenger-Satisfaction-Analysis-Project (Yolcu Memnuniyeti Analizi Projesi)


Bu proje, bir hava yolu şirketinin müşteri memnuniyetini analiz etmek için gerçekleştirilmiştir. İki aşamalı bir süreci kapsar: veri analizi ve makine öğrenimi algoritmalarının uygulanması. İlk aşamada, hava yolu şirketinin yolcu verileri incelenir ve temizlenir. İkinci aşamada ise, temizlenmiş veri üzerinde çeşitli makine öğrenimi algoritmaları kullanılarak müşteri memnuniyeti tahmini yapılır.

## Proje Aşamaları

### Veri Temizleme ve Hazırlama

- **Eksik Veri Doldurma**: Eksik veriler, ilgili sütunların ortalamasıyla doldurulmuştur.
- **Gereksiz Sütunların Silinmesi**: Analiz için gereksiz olduğu belirlenen sütunlar veri setinden çıkarılmıştır.
- **Kategorik Değişkenlerin Etiketlenmesi**: Makine öğrenimi algoritmaları için gerekli olan kategorik değişkenler sayısal değerlere dönüştürülmüştür.

### Veri Analizi

- **Korelasyon Matrisi Çıkarılması**: Veri setindeki değişkenler arasındaki ilişkileri görselleştirmek için bir korelasyon matrisi oluşturulmuştur.
- **Tekrar Eden Satırların Bulunması**: Veri setindeki tekrar eden satırlar tespit edilmiş ve raporlanmıştır.

### Veri Görselleştirme

- **Keman Grafiği Çizimi**: Müşteri memnuniyeti ile yaş, cinsiyet gibi değişkenler arasındaki ilişkiyi gösteren keman grafiği çizilmiştir.
- **Histogramlar**: Veri setindeki değişkenlerin dağılımını gösteren histogramlar oluşturulmuştur.

### Makine Öğrenimi Algoritmalarının Uygulanması

- **Neural Network (Yapay Sinir Ağı)**: Yapay Sinir Ağı modeli, veri seti üzerinde eğitilmiş ve müşteri memnuniyeti tahmini için kullanılmıştır.
- **Rastgele Orman (Random Forest)**: Veri seti üzerinde rastgele orman algoritması uygulanmıştır.
- **Lojistik Regresyon**: Temel bir sınıflandırma modeli olan lojistik regresyon kullanılarak müşteri memnuniyeti tahmini yapılmıştır.
- **Karar Ağacı (Decision Tree)**: Karar ağacı algoritması ile model eğitilmiş ve sonuçlar değerlendirilmiştir.
- **K-En Yakın Komşu (KNN)**: KNN algoritması ile model eğitilmiş ve performansı ölçülmüştür.
- **XGBoost**: Güçlü bir sınıflandırma algoritması olan XGBoost kullanılarak model eğitilmiştir.
- **Gradient Boosting (Gradyan Artırma)**: Gradyan Artırma yöntemi, zayıf öğrenicilerin (genellikle karar ağaçları) bir araya getirilerek güçlü bir öğrenici oluşturulmasını sağlar. Bu çalışmada, Gradyan Artırma sınıflandırma algoritması, müşteri memnuniyetini tahmin etmek için kullanılmış ve modelin performansını artırmak için veri setindeki karmaşıklığı ele almak için etkili bir yöntem olarak benimsenmiştir.
- **Gaussian Naive Bayes (Gauss Naif Bayes)**: Gauss Naif Bayes sınıflandırma algoritması, sınıflandırma problemlerinde sıkça kullanılan basit ancak etkili bir olasılık tabanlı yöntemdir. Bu çalışmada, Gauss Naif Bayes algoritması kullanılarak müşteri memnuniyeti tahmini gerçekleştirilmiştir. Naif Bayes'in temel varsayımları nedeniyle, veri setindeki özellikler arasındaki bağımsızlık varsayımı gereği bu algoritmanın performansı bazı durumlarda diğer karmaşık modellere kıyasla daha düşük olabilir. Ancak, bazı uygulamalarda hızlı eğitim ve tahmin avantajları sağlayabilir.
  
## Proje Çalıştırma Adımları

1. **Gerekli Kütüphanelerin Kurulumu**: Projeyi çalıştırmak için gerekli olan kütüphaneleri `requirements.txt` dosyasını kullanarak kurun.
    ```
    pip install -r requirements.txt
    ```

2. **Veri Setlerinin İndirilmesi**: Projeyi çalıştırmak için gerekli olan eğitim ve test veri setlerini sağlayan bağlantılardan indirin ve uygun bir dizine kaydedin.

3. **Ana Kodun Çalıştırılması**: `proje.py` dosyasını çalıştırarak projeyi başlatın.
    ```
    python proje.py
    ```

## Sonuçlar:
   
![result](https://github.com/SemihGul5/MachineLearning-Passenger-Satisfaction-Analysis-Project/assets/133046330/c673679a-f337-4b03-ae14-948bc5277f6b)

 Projeden elde edilen ayrıntılı sonuçları içeren çıktıyı `rapor`'da bulabilirsiniz.

## Katkı Adımları

1. GitHub'dan projeyi klonlayın veya çatallayın.
2. Yeni bir dal oluşturun ve değişikliklerinizi yapın.
3. Değişikliklerinizi kaydedin ve ana depoya gönderin.
4. Bir çekme isteği oluşturun ve proje sahibine gönderin.
5. Çekme isteğinin incelenmesini bekleyin ve kabul edilmesini bekleyin.

## Kaynaklar

Bu projenin hazırlanmasında aşağıdaki kaynaklar kullanılmıştır:

- [Kaggle - Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Plotly Documentation](https://plotly.com/python/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
