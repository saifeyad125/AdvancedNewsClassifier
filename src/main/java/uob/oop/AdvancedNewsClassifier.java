package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        List<double[]> vector = Toolkit.getlistVectors();
        for (int i = 0; i < Toolkit.listVocabulary.size(); i++){
            String word = Toolkit.listVocabulary.get(i);
            if (!(isStopWord(Toolkit.STOPWORDS, word))){
                 double[] lis = vector.get(i);
                 Vector v = new Vector(lis);
                 Glove g = new Glove(Toolkit.listVocabulary.get(i), v);
                 listResult.add(g);
                 }
            }
        return listResult;
    }

    public boolean isStopWord(String[] sWords, String curWord){
        for (String word : sWords){
            if (word.equals(curWord)){
                return true;
            }
        }
        return false;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }


    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        List<Integer> lengths = new ArrayList<>();
        for (ArticlesEmbedding article : _listEmbedding){
            int size = articleSize(article.getNewsContent());
            lengths.add(size);
        }
        merge(lengths);
        if ((lengths.size()%2) == 0){
            int half = lengths.size()/2;
            intMedian = (lengths.get(half) + lengths.get(half + 1)) /2;
        }
        //so if odd
        else{
            int half = (lengths.size()+1)/2;
            intMedian = lengths.get(half);
        }
        return intMedian;
    }

   public int articleSize(String article) {
       String[] words = article.split(" ");
       int count = 0;
       for (String word : words) {
           for (Glove g : listGlove) {
               if (word.equals(g.getVocabulary())) {
                   count++;
                   break;
               }
           }
       }
       return count;
   }


   //mergesort here
   public static void merge(List<Integer> list){
       if (list.size() < 2){
           return;
       }
       int mid = list.size()/2;
       List<Integer> left = new ArrayList<>(list.subList(0, mid));
       List<Integer> right = new ArrayList<>(list.subList(mid, list.size()));
       merge(left);
       merge(right);
       mergehelp(left, right, list);

   }
    public static void mergehelp(List<Integer> left, List<Integer> right, List<Integer> list){
        int l = 0, r = 0;
        list.clear();
        while (l < left.size()&& r < right.size()){
            if (left.get(l) < right.get(r)){
                list.add(left.get(l));
                l++;
            }
            else{
                list.add(right.get(r));
                r++;
            }

        }
        while (l<left.size()){
            list.add(left.get(l));
            l++;
        }
        while (r<right.size()){
            list.add(right.get(r));
            r++;
        }
    }

    public void printLabels(){
        for (NewsArticles article : listNews){
            if (article.getNewsType() == NewsArticles.DataType.Training) {
                System.out.println(article.getNewsTitle());
            }
        }
    }





    public void populateEmbedding() {
        //loop through each article and get embedding
        //
        for (ArticlesEmbedding article: listEmbedding){
            boolean worked = false;
            while (!worked){
                try{
                    article.getEmbedding();
                    worked = true;

                } catch (InvalidSizeException e) {
                    //if intSize is -1 we would set intSize to its actual value
                    article.setEmbeddingSize(embeddingSize);
                }
                catch (InvalidTextException e){
                    article.getNewsContent();
                }
                catch (Exception e){
                    throw new RuntimeException(e);
                }
            }

        }

    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        /*
        each DataSet object has two elements, an input, and an output (INDArray)
        the input array is just the document level embedding .getEmbedding()
        the shape of the output array is: [1, _numberOfClasses]
         */
        for (ArticlesEmbedding article : listEmbedding){
            if (article.getNewsType() == NewsArticles.DataType.Training){
                inputNDArray = article.getEmbedding();
                outputNDArray = Nd4j.zeros(1, _numberOfClasses);
                int INTLabel = Integer.parseInt(article.getNewsLabel());
                //I will check if the label is smaller than the number of groups, if it is then I'll add it
                if (INTLabel <= _numberOfClasses && INTLabel > 0) {
                    //here I'll add the value based on the label, if it's
                    outputNDArray.putScalar(new int[]{0, INTLabel-1}, 1);
                    DataSet myDataSet = new DataSet(inputNDArray, outputNDArray);
                    listDS.add(myDataSet);
                }
            }
        }




        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();

        for (ArticlesEmbedding article: _listEmbedding){
            if (article.getNewsType() == NewsArticles.DataType.Testing){
                int[] labelList = myNeuralNetwork.predict(article.getEmbedding());
                int label = labelList[0];
                listResult.add(label);
                label++;
                article.setNewsLabel(String.valueOf(label));
            }
        }
        return listResult;
    }

    public void printResults() {

        int count = 0;
        List<Integer> labels = new ArrayList<>();
        labels.add(1);
        boolean finished = false;
        while (!finished) {
            int curLabel = labels.get(count);
            System.out.println("Group " + curLabel);
            for (ArticlesEmbedding article : listEmbedding) {
                if (article.getNewsType() == NewsArticles.DataType.Testing) {
                    int label = Integer.parseInt(article.getNewsLabel());
                    if (!(labels.contains(label))) {
                        labels.add(label);
                    } else if (curLabel == label) {
                        System.out.println(article.getNewsTitle());
                    }
                }

            }
            if (count == labels.size()-1) {
                finished = true;
            }
            count++;

        }
    }





}
