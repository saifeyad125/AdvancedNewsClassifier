package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        super(_title,_content,_type,_label);

    }

    public void setEmbeddingSize(int _size) {
        intSize = _size;

    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        if (processedText.isEmpty()) {
            String text = textCleaning(super.getNewsContent());
            text = lemmatize(text);
            processedText = SW(text, Toolkit.STOPWORDS).toLowerCase();
            return processedText.trim();

        }
        return processedText.trim();


    }


    public String lemmatize(String text){
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,pos,lemma");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = pipeline.processToCoreDocument(text);
        StringBuilder sb = new StringBuilder();
        // display tokens
        for (CoreLabel tok : document.tokens()) {
            sb.append(tok.lemma()).append(" ");
        }
        return sb.toString();
    }



    public String SW(String _content, String[] _stopWords){
        StringBuilder sbContent = new StringBuilder(_content.length());
        String[] words = _content.split(" ");
        boolean found;
        for (String w : words) {
            found = false;
            for (String cur : _stopWords) {
                if (w.equals(cur)) {
                    found = true;
                    break;
                }
            }
            if (!found) sbContent.append(w).append(" ");
        }

        return sbContent.toString().trim();
    }





    public INDArray getEmbedding() throws Exception {
        /*
        processed text to words
        get lower num between intSize and contents.length
        create NDFJ array full of 0 and intSize as num of rows and vectorSize as num of columns
        counter for how many words u put into array
        loop through the glove if word == glove then u create IND array holds vectors using glove.getelements
        after for loop of glvoes, check if count
        if count then break?
         */
        if (intSize == -1){
            throw new InvalidSizeException("Invalid size");
        }
        if (processedText.isEmpty()) {
            throw new InvalidTextException("Invalid text");}

        //first I will get glove list
        if (newsEmbedding.isEmpty()) {
            Glove[] gloves = new Glove[AdvancedNewsClassifier.listGlove.size()];
            int count = 0;
            for (Glove g : AdvancedNewsClassifier.listGlove) {
                gloves[count] = g;
                count++;
            }
            //making it all 0s and the x is intSize and y is the vector size (assuming each word has the same vector size
            newsEmbedding = Nd4j.zeros(intSize, gloves[0].getVector().getVectorSize());
            String[] words = processedText.split(" ");
            //this is to make it more efficient either it reaches the end of the array or reaches the max value
            int maxEmbedding = Math.min(intSize, words.length);
            //now I'll loop through the glove and words, if it has a glove object then I add it
            count = 0;
            for (String W : words) {
                if (count >= intSize) break;
                for (Glove g : gloves) {
                    if (W.equals(g.getVocabulary())) {
                        //we add it to the array
                        INDArray cur = Nd4j.create(g.getVector().getAllElements());
                        newsEmbedding.putRow(count, cur);
                        count++;
                        break;
                    }
                }
            }
        }







        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
