# -------------------- ΒΗΜΑ 1: ΠΡΟΕΤΟΙΜΑΣΙΑ - Ενεργοποίηση των βιβλιοθηκών --------------------
library(tm)
library(SnowballC)
library(topicmodels)
library(syuzhet)
library(ggplot2)
library(wordcloud)
library(topicdoc)
library(slam)
library(tidytext)
library(dplyr)
library(reshape2)
# -------------------- ΒΗΜΑ 2: ΕΠΙΛΟΓΗ ΤΟΥ ΘΕΜΑΤΟΣ ΚΑΙ ΦΟΡΤΩΣΗ ΤΩΝ ΔΕΔΟΜΕΝΩΝ --------------------
path <- "covid19_tweets.csv"# Ορισμός της διαδρομής του αρχείου (Path)

my_data <- read.csv(path, stringsAsFactors = FALSE) # Φόρτωση των δεδομένων στην R - Το stringsAsFactors = FALSE είναι πολύ σημαντικό για να διαβάζει τα κείμενα ως "λέξεις" και όχι ως "κατηγορίες"

View(my_data) # Προεπισκόπηση των δεδομένων για το αν φορτώθηκαν οι στήλες

nrow(my_data) # Πόσα tweets έχουμε;

names(my_data) # Ποιες είναι οι στήλες; 
# -------------------- ΒΗΜΑ 3: ΚΑΘΑΡΙΣΜΟΣ ΤΩΝ ΔΕΔΟΜΕΝΩΝ (PREPROCESSING) --------------------
# 1. Δημιουργία Corpus (Σώμα Κειμένων) - Μετατρέπουμε τα tweets σε μια μορφή που η βιβλιοθήκη 'tm' μπορεί να επεξεργαστεί
docs <- Corpus(VectorSource(my_data$text))

# 2. Μετατροπή σε πεζά γράμματα (Lowercase)
docs <- tm_map(docs, content_transformer(tolower))

# 3. Αφαίρεση URLs (Links) - Απαραίτητο για δεδομένα από Twitter
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
docs <- tm_map(docs, content_transformer(removeURL))

# 3.1 Χειροκίνητη αφαίρεση ειδικών χαρακτήρων (όπως το … και το amp)
# Χρησιμοποιούμε την gsub για να αντικαταστήσουμε αυτούς τους χαρακτήρες με κενό
docs <- tm_map(docs, content_transformer(function(x) gsub("…", " ", x)))
docs <- tm_map(docs, content_transformer(function(x) gsub("amp", " ", x)))
docs <- tm_map(docs, content_transformer(function(x) gsub("covid…", "covid", x)))

# 4. Αφαίρεση σημείων στίξης (Punctuation)
docs <- tm_map(docs, removePunctuation)

# 5. Αφαίρεση αριθμών (Numbers)
docs <- tm_map(docs, removeNumbers)

# 6. Αφαίρεση stopwords (Κοινές λέξεις χωρίς νόημα) - Χρησιμοποιούμε την αγγλική λίστα αφού τα tweets είναι στα Αγγλικά
docs <- tm_map(docs, removeWords, stopwords("english"))

# 7. Stemming (Αναγωγή στη ρίζα) - 4η Διάλεξη, σελ. 11 - Π.χ. το "fishing", "fished", "fisher" γίνονται όλα "fish"
docs <- tm_map(docs, stemDocument)

# 8. Αφαίρεση περιττών κενών διαστημάτων (Strip Whitespace)
docs <- tm_map(docs, stripWhitespace)

# 9. Tokenization & Δημιουργία Document-Term Matrix (DTM) - Εδώ το κείμενο "σπάει" σε tokens (λέξεις) και μετατρέπεται σε πίνακα
dtm <- DocumentTermMatrix(docs)

# 10. Βελτιστοποίηση: Αφαίρεση σπάνιων όρων (Sparsity) - Κρατάμε τις λέξεις που εμφανίζονται σε τουλάχιστον 1% των εγγράφων για να μην είναι ο πίνακας υπερβολικά τεράστιος
dtm_cleaned <- removeSparseTerms(dtm, 0.99)

# 11. Τελικός Έλεγχος
writeLines(as.character(docs[[1]])) # Πώς μοιάζει το πρώτο tweet μετά από όλη τη διαδικασία
# Οι διαστάσεις του πίνακα DTM (Έγγραφα x Λέξεις)
dim(dtm) # χωρίς parsing
dim(dtm_cleaned) # με parsing
# -------------------- ΒΗΜΑ 4: TOPIC MODELING (LDA) --------------------
# 1. Διαχείριση κενών εγγραφών - Πριν το LDA, αφαιρούμε tweets που έμειναν χωρίς λέξεις μετά τον καθαρισμό
row_totals_full <- row_sums(dtm)
dtm_full <- dtm[row_totals_full > 0, ]

row_totals_clean <- apply(dtm_cleaned, 1, sum)
dtm_limited <- dtm_cleaned[row_totals_clean > 0, ]

# 2. ΕΥΡΕΣΗ ΒΕΛΤΙΣΤΟΥ K (PERPLEXITY & COHERENCE)
# Προετοιμασία πίνακα αποτελεσμάτων
k_values <- c(2, 3, 5, 7, 9, 11, 13, 15, 16)
results <- data.frame(k = k_values, perplexity = NA, coherence = NA)

for(i in 1:length(k_values)) { # 1. Εκπαίδευση του μοντέλου (Μία φορά για κάθε k)
  temp_lda <- LDA(dtm_limited, k = k_values[i], method = "Gibbs", 
                  control = list(iter = 500, seed = 42))
  
  results$perplexity[i] <- topicmodels::perplexity(temp_lda, dtm_limited) # 2. Υπολογισμός Perplexity (Θέλουμε ΦΘΙΝΟΥΣΑ πορεία)
  
  results$coherence[i] <- mean(topicdoc::topic_coherence(temp_lda, dtm_limited)) # 3. Υπολογισμός Coherence (Θέλουμε ΑΥΞΟΥΣΑ πορεία/Κορυφή)
  
  print(paste("Ολοκληρώθηκε η ανάλυση για k =", k_values[i]))
}

# --- ΓΡΑΦΗΜΑΤΑ ΑΞΙΟΛΟΓΗΣΗΣ ---
# Γράφημα i: Perplexity (Elbow Method)
ggplot(results, aes(x = k, y = perplexity)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 3) +
  labs(title = "Αξιολόγηση μέσω Perplexity (Lower is Better)",
       x = "Αριθμός Θεμάτων (k)", y = "Perplexity") +
  theme_minimal()

# Γράφημα ii: Coherence (Higher is Better)
ggplot(results, aes(x = k, y = coherence)) +
  geom_line(color = "darkgreen", size = 1) +
  geom_point(color = "orange", size = 3) +
  labs(title = "Αξιολόγηση μέσω Coherence Score (Higher is Better)",
       x = "Αριθμός Θεμάτων (k)", y = "Mean Coherence") +
  theme_minimal()

# 3. Εκτέλεση του μοντέλου LDA - Ορίζουμε k = 5 θέματα 
lda_limited <- LDA(dtm_limited, k = 5, method = "Gibbs", control = list(iter = 1000, seed = 42, alpha = 0.1, delta = 0.1))
require(topicmodels)
terms(lda_limited, 10) # Εξαγωγή των 10 σημαντικότερων λέξεων ανά θέμα
topics(lda_limited)[1:10] # Δες πού ανήκουν τα πρώτα 10 tweets

# 4. Απεικόνιση (ερώτημα 3 - ii)
# Ραβδόγραμμα ----------
top_terms <- tidy(lda_limited, matrix = "beta") %>% # 1. Προετοιμασία δεδομένων
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

# 2. Σχεδίαση
ggplot(top_terms, aes(beta, reorder_within(term, beta, topic), fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered() +
  labs(title = "Κορυφαίες 10 λέξεις ανά θεματική ενότητα (k=5)",
       subtitle = "Beta Scores - Πιθανότητα εμφάνισης όρων",
       x = "Πιθανότητα (Beta Score)",
       y = "Λέξεις") +
  theme_minimal()

# Συννεφάκια λέξεων ----------
par(mfrow=c(2,3)) 

for(i in 1:5) { # Φιλτράρουμε τις λέξεις για το θέμα i
  topic_words <- tidy(lda_limited, matrix = "beta") %>%
    filter(topic == i) %>%
    arrange(desc(beta)) %>%
    head(50) 
  
  # Δημιουργία Word Cloud
  wordcloud(words = topic_words$term, 
            freq = topic_words$beta, 
            scale = c(4, 1), # Προσαρμογή μεγέθους λέξεων
            max.words = 50, 
            random.order = FALSE, 
            colors = brewer.pal(8, "Dark2"))
  
  # Προσθήκη τίτλου πάνω από κάθε σύννεφο
  title(main = paste("Θέμα", i), line = -1)
}

# Επαναφορά της ρύθμισης του παραθύρου σε 1x1
par(mfrow = c(1, 1))
# -------------------- ΒΗΜΑ 5: ΑΝΑΛΥΣΗ ΣΥΝΑΙΣΘΗΜΑΤΟΣ & MATCHING --------------------
cat("Αρχικά Tweets:", nrow(my_data), "\n") # Για μια γρήγορη σύγκριση με τον αρχικό αριθμό
cat("Tweets μετά το φιλτράρισμα:", nrow(results_df), "\n")
cat("Tweets που αφαιρέθηκαν (ήταν κενά):", nrow(my_data) - nrow(results_df), "\n")
# 1. Αποθήκευση του Matching (Ποιο tweet ανήκει σε ποιο topic) - Χρησιμοποιούμε το dtm_limited γιατί αυτό τρέξαμε στο LDA.
results_df <- my_data[row_totals_clean > 0, ] # φιλτραρισμένο αντίγραφο των δεδομένων από τα tweets που περιείχαν τουλάχιστον μία «χρήσιμη» λέξη μετά τον καθαρισμό
results_df$topic <- topics(lda_limited) # Δημιουργεί μια νέα στήλη με όνομα topic μέσα στον results_df που λεει ποιο tweet ειναι σε ποιο topic

# 2. "Μερικός Καθαρισμός" για το Sentiment Analysis - Παίρνουμε το κείμενο των tweets που αντιστοιχούν στο LDA
sentiment_text <- results_df$text
# Κάνουμε μόνο τα απαραίτητα
sentiment_text <- tolower(sentiment_text) # Μετατροπή σε πεζά γράμματα (Lowercase)
sentiment_text <- gsub("http[^[:space:]]*", "", sentiment_text) # Αφαίρεση των URLs
sentiment_text <- gsub("[[:punct:]]", " ", sentiment_text) # Αφαίρεση σημείων στίξης
sentiment_text <- gsub("[[:digit:]]", "", sentiment_text) # Αφαίρεση των αριθμών
sentiment_text <- stripWhitespace(sentiment_text) # Καθαρισμός περιττών κενών

# 3. Ειδικά Συναισθήματα (Anger, Joy κλπ) από NRC (Επίπεδο β)
emotions_nrc <- get_nrc_sentiment(sentiment_text)

# 4. # Χρήση της ομώνυμης μεθόδου "syuzhet" που δίνει δεκαδικά scores
decimal_scores <- get_sentiment(sentiment_text, method = "syuzhet")

# 5. Ένωση όλων στον τελικό πίνακα - Συνδυάζουμε τα tweets, τα topics, τα 8 συναισθήματα και το σκορ AFINN
final_table <- cbind(results_df, emotions_nrc) # Εδώ κολλάμε τα συναισθήματα NRC δίπλα στα δεδομένα μας
final_table$sentiment_score <- decimal_scores # Προσθήκη του δεκαδικού score στον final_table από τη μέθοδο syuzhet

# 6. Κατηγοριοποίηση με "Ζώνη Ουδετερότητας" - Ορίζουμε ότι από -0.15 έως 0.15 το tweet είναι Neutral
final_table$sentiment_type <- ifelse(final_table$sentiment_score > 0.15, "Positive",
                                     ifelse(final_table$sentiment_score < -0.15, "Negative", "Neutral"))

# 7. Ονοματοδοσία των Topics (Για να φαίνονται σωστά στα γραφήματα)
final_table$topic_name <- factor(final_table$topic,
                                 levels = 1:5,
                                 labels = c("Prevention & Masks", 
                                            "Politics & Public Opinion", 
                                            "Science & Vaccines", 
                                            "Statistics & Global Cases", 
                                            "Healthcare & Social Impact"))

# 8. Έλεγχος του τελικού πίνακα για να δούμε τις νέες στήλες
View(head(final_table, 20))

# 8. Πόσα tweets μπήκαν σε κάθε κατηγορία
table(final_table$sentiment_type)

# 9. Απεικόνιση
# i) Grouped Bar Chart
ggplot(final_table, aes(x = topic_name, fill = sentiment_type)) +
  geom_bar(position = "dodge") + # Το "dodge" βάζει τις μπάρες δίπλα-δίπλα
  coord_flip() + # Αυτή η εντολή αντιστρέφει τους άξονες X και Y
  scale_fill_manual(values = c("Negative" = "#E41A1C", 
                               "Neutral" = "#999999", 
                               "Positive" = "#4DAF4A")) +
  labs(title = "Sentiment Counts per Topic",
       subtitle = "Side-by-side comparison of sentiment categories across topics",
       x = "Topic Name",
       y = "Number of Tweets",
       fill = "Sentiment Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ii) NRC Emotions Bar Chart
# 1. Υπολογισμός των συνόλων για κάθε συναίσθημα
# Παίρνουμε μόνο τις 8 στήλες των συναισθημάτων από τον final_table
emotion_sums <- colSums(final_table[, c("anger", "anticipation", "disgust", "fear", 
                                        "joy", "sadness", "surprise", "trust")])

# 2. Μετατροπή σε dataframe για το ggplot
emotion_df <- data.frame(emotion = names(emotion_sums), count = emotion_sums)

# 3. Δημιουργία του γραφήματος
ggplot(emotion_df, aes(x = reorder(emotion, count), y = count, fill = emotion)) +
  geom_bar(stat = "identity") +
  # Προσθήκη των αριθμών
  geom_text(aes(label = count), 
            hjust = 1.2,          # Τοποθέτηση λίγο μέσα από την άκρη της μπάρας
            color = "white",      # Λευκό χρώμα για να κάνει αντίθεση με τη μπάρα
            size = 4,             # Μέγεθος γραμματοσειράς
            fontface = "bold") +  # Έντονα γράμματα
  coord_flip() + # Οριζόντιο για ευκολία στην ανάγνωση
  labs(title = "Total Sentiment Profile (NRC)",
       subtitle = "Frequency of 8 basic emotions across all tweets",
       x = "Emotion",
       y = "Total Score (Intensity)") +
  theme_minimal() +
  guides(fill = "none") # Αφαιρούμε τη λεζάντα γιατί τα ονόματα είναι ήδη στον άξονα

# iii) Comparison Word Cloud
# 1. Δημιουργούμε δύο "κουβάδες" κειμένου: έναν για Positive και έναν για Negative
pos_text <- paste(final_table$text[final_table$sentiment_type == "Positive"], collapse = " ")
neg_text <- paste(final_table$text[final_table$sentiment_type == "Negative"], collapse = " ")

# 2. Τα ενώνουμε σε ένα διάνυσμα
all_sent_text <- c(pos_text, neg_text)

# 3. Δημιουργούμε ένα Term-Matrix (Συχνότητα λέξεων ανά κατηγορία)
# Χρησιμοποιούμε τον καθαρισμό που μάθαμε
all_corpus <- Corpus(VectorSource(all_sent_text))
all_tdm <- TermDocumentMatrix(all_corpus, control = list(
  removePunctuation = TRUE,
  stopwords = stopwords("english"),
  removeNumbers = TRUE,
  tolower = TRUE
))

# 4. Μετατροπή σε matrix και ονοματοδοσία στηλών
all_m <- as.matrix(all_tdm)
colnames(all_m) <- c("Positive", "Negative")

# 5. Σχεδίαση του Comparison Cloud
comparison.cloud(all_m, 
                 colors = c("#4DAF4A", "#E41A1C"), # Πράσινο για Positive, Κόκκινο για Negative
                 max.words = 100,                  # Οι 100 πιο σημαντικές λέξεις
                 title.size = 2, 
                 scale = c(3, 0.5))                # Μέγεθος λέξεων

# iv) Scatter Plot.
# Παίρνουμε ένα δείγμα 10.000 tweets για να μην κολλήσει το γράφημα (είναι πολλά τα 170k για scatter plot)
set.seed(123)
sample_data <- final_table[sample(nrow(final_table), 10000), ]

ggplot(sample_data, aes(x = topic_name, y = sentiment_score, color = topic_name)) +
  geom_jitter(alpha = 0.4, size = 1.5) + # Δημιουργεί το εφέ του "σύννεφου"/cluster
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") + # Γραμμή ουδετερότητας
  labs(title = "Topic Clusters by Sentiment Intensity",
       subtitle = "Visualizing tweet distribution across topics and sentiment scores",
       x = "Topic",
       y = "Sentiment Score (Syuzhet)",
       color = "Topic") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# v) Top Words per Topic Bar Chart
# 1. Εξαγωγή των λέξεων (β-scores) από το μοντέλο LDA
topics_words <- tidy(lda_limited, matrix = "beta")

# 2. Επιλογή των 10 κορυφαίων λέξεων για κάθε topic
top_terms <- topics_words %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>% 
  ungroup() %>%
  arrange(topic, -beta)

# 3. Προσθήκη των ονομάτων που δώσαμε στα Topics
top_terms$topic_name <- factor(top_terms$topic,
                               levels = 1:5,
                               labels = c("Prevention & Masks", 
                                          "Politics & Opinion", 
                                          "Science & Vaccines", 
                                          "Statistics & Global", 
                                          "Healthcare & Impact"))

# 4. Σχεδίαση του γραφήματος
ggplot(top_terms, aes(beta, reorder_within(term, beta, topic_name), fill = topic_name)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic_name, scales = "free", ncol = 2) + # Χωρίζει το γράφημα σε 5 μικρότερα
  scale_y_reordered() +
  labs(title = "Top 10 Terms per Topic",
       subtitle = "Highest probability words (beta) for each COVID-19 theme",
       x = "Beta (Word Probability)",
       y = NULL) +
  theme_minimal()