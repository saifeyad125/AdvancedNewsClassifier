package uob.oop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        listVocabulary = new ArrayList<>();
        listVectors = new ArrayList<>();
        BufferedReader myReader = null;
        try {
            File path = Toolkit.getFileFromResource(FILENAME_GLOVE);
            FileReader filer = new FileReader(path);
            myReader = new BufferedReader(filer);
            String[] lis;
            String wordline;
            //i'll split each line into a list then see if it's the name or vector
            while ((wordline = myReader.readLine()) != null) {
                lis = wordline.split(",");
                listVocabulary.add(lis[0]);
                double[] doubleVecArr = new double[lis.length - 1];
                //I will make the vector array here
                for (int i = 1; i < lis.length; i++) {
                    doubleVecArr[i - 1] = Double.parseDouble(lis[i]);
                }
                listVectors.add(doubleVecArr);
            }


        } catch (IOException | URISyntaxException e) {
            e.getMessage();
        } finally {
            assert myReader != null;
            myReader.close();
        }


    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        File[] f = new File("src/main/resources/News").listFiles();
        assert f != null;
        mergeFile(f);
        for (File file : f) {
            String ending = file.getName();
            ending = ending.substring(ending.length() - 4);
            if (ending.equals(".htm")) {
                try {
                    String body = Files.readString(file.toPath());
                    String title = HtmlParser.getNewsTitle(body);
                    String content = HtmlParser.getNewsContent(body);
                    NewsArticles.DataType type = HtmlParser.getDataType(body);
                    String label = HtmlParser.getLabel(body);
                    NewsArticles a = new NewsArticles(title, content, type, label);
                    listNews.add(a);
                } catch (IOException e) {
                    System.out.println("IOException.");
                }
            }
        }

        return listNews;
    }


        public static void mergeFile(File[] list){
            if (list.length < 2){
                return;
            }

            int mid = list.length/2;
            File[] left = new File[mid];
            File[] right = new File[list.length-mid];
            for (int i = 0; i < mid; i++){
                left[i] = list[i];
            }
            for (int i = mid; i < list.length; i++){
                right[i-mid] = list[i];
            }
            mergeFile(left);
            mergeFile(right);
            mergehelp(left, right, list);
        }
        public static void mergehelp(File[] left, File[] right, File [] list){
            int l = 0, r = 0, k = 0;
            while (l < left.length && r < right.length){
                String leftValS = left[l].getName().substring(0, 2);
                String rightValS = right[r].getName().substring(0, 2);
                int leftVal = Integer.parseInt(leftValS);
                int rightVal = Integer.parseInt(rightValS);
                if (leftVal < rightVal){
                    list[k] = left[l];
                    l++;
                }
                else{
                    list[k] = right[r];
                    r++;
                }
                k++;
            }
            while (l<left.length){
                list[k] = left[l];
                l++;
                k++;
            }
            while (r<right.length){
                list[k] = right[r];
                r++;
                k++;
            }
        }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
