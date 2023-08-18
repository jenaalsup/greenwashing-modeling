# load libraries
library(tidytext)
library(dplyr)
library(tidyverse)
library(quanteda)
library(parallel)
library(stm)
library(tm)
library(tmtoolkit)
library(ggplot2)

# load data
textData <- read.csv("reports/processed_reports/processed-data.csv")
textData <- textData[!duplicated(textData$doc_id),]
capex_data <- read_csv("reports/processed_reports/capex-data.csv")
textData <- textData %>%left_join(capex_data)

# prepare JST model
attach(textData)
llDisplay <- data.frame(doc_id = paste(doc_id),text = paste(text),company_type=paste(company_type),company_ticker=paste(company_ticker),year=paste(year),part=part,stringsAsFactors = F)
detach(textData)

# add custom stop words
stop_words_custom = c("will","has","by","for","hi","hey","are","as","i","we","our","ours","ourselves","use",
                       "you","your","yours","that","f","e","s","t","c","n","u","v","l","p","d","b","g","k","m","x","y","z",
                       "be","with","is","was","been","not","they","way","and","to","do","go","on","have","from",
                       "at","but","or","an","if","all","so","it","thing","put","well","take","see","","can't","can",
                       "got","cant","could","him","his","this","had","he","her","she","hers","their","they're","things",
                       "go","going","let","would","make","like","come","us",paste(textData$firstname),paste(tidytext::stop_words$word),stopwords())

# create tokens for kept words
toks1 <- tokens(corpus(llDisplay), remove_punct = TRUE)
toks2 <- tokens_remove(toks1, stop_words_custom)
toks3 <- tokens_ngrams(toks2, 1:2) # was 1:3
toks3 <- tokens_wordstem(toks3)

# create a document feature matrix (quanteda style) to run jst
N = nrow(llDisplay)
dfm_speeches <- dfm(toks3, remove_punct = TRUE, remove_numbers = TRUE, stem = T) %>% 
  dfm_trim(min_termfreq = 15, min_docfreq =0.01*N, max_docfreq = 0.8*N) # was 25, 0.002, 0.4

dfm_speeches <- dfm_subset(dfm_speeches,ntoken(dfm_speeches) >= 1)

# prep data for stm
houseDfmSTM <- convert(dfm_speeches, to = "stm", 
                       docvars = docvars(dfm_speeches))

documents <- houseDfmSTM$documents
vocab     <- houseDfmSTM$vocab
meta      <- houseDfmSTM$meta

# select number of topics
kk = c(5,10,15,20,25,28,30,40,50,60)

# run stm with metadata on prevalence 
model_stm <- mclapply(kk,
                      FUN = function(kk) 
                        stm(houseDfmSTM$documents, 
                            houseDfmSTM$vocab,
                            prevalence=~factor(company_type) + factor(year),
                            K = kk,
                            max.em.its = 200,
                            data = houseDfmSTM$meta, 
                            init.type = "Spectral",
                            seed = 153332),
                            mc.cores = getOption("mc.cores", 5L))

save(model_stm,file="modelOutput-House-STM2.RData")

# view topic stats
labelTopics(model_stm[[4]]) # FREX = frequent + not shared by other topics

# calculate umass coherence
coherence_scores <- lapply(model_stm, function(model) {
  semanticCoherence(model, documents)
})
average_coherence_per_model <- sapply(coherence_scores, mean)
print(average_coherence_per_model)

# topic selection: 4th model, 20 topics, topic 5 & 14
model_stm[[4]]$theta # doc-topic matrix - row is doc and column is probability it is in that topic
model_stm[[4]]$theta[, 5]
model_stm[[4]]$theta[, 14]

# after topic selection
model_stm[[4]]$settings$covariates # covariates 

# correlate the green columns of theta (the topics) with the capex data
# METHOD 1:
topic_proportions <- model_stm[[4]]$theta
capex_percentages = round(100 * textData$env_capex / textData$capex, digits = 0)
num_topics <- ncol(model_stm[[4]]$theta)
correlations <- numeric(num_topics)
for (topic_idx in 1:num_topics) {
  topic_proportions <- model_stm[[4]]$theta[, topic_idx]
  correlations[topic_idx] <- cor(topic_proportions, capex_percentages)
}
# METHOD 2:
num_documents <- nrow(textData)
num_topics <- ncol(model_stm[[4]]$theta)
correlations <- matrix(0, nrow = num_documents, ncol = num_topics)

for (doc_idx in 1:num_documents) {
  for (topic_idx in 1:num_topics) {
    topic_proportions <- model_stm[[4]]$theta[doc_idx, topic_idx]
    capex_percentage <- capex_percentages[doc_idx]
    
    # Check if either value is zero before calculating the correlation
    if (!is.na(topic_proportions) && !is.na(capex_percentage) && topic_proportions != 0 && capex_percentage != 0) {
      correlations[doc_idx, topic_idx] <- cor(topic_proportions, capex_percentage)
    }
  }
}
print(correlations) # TODO: all 0 for energy, all NA for renewable energy

# each dot is a company, plot the probability that a company's documents appear in a selected topic against the env capex percentage for that company
# TODO: import actual data here
company_data <- data.frame(
  Company = c("agr", "arry", "cop", "cvx", "cwen"),
  Topic_Probability = c(0.8, 0.6, 0.4, 0.7, 0.9),
  Env_Capex_Percentage = c(15, 20, 10, 25, 30)
)

ggplot(company_data, aes(x = Env_Capex_Percentage, y = Topic_Probability)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) + # method = "lm" adds regression, se = TRUE adds confidence interval
  labs(title = "Topic Probability vs. Env Capex Percentage",
       x = "Environmental Capex Percentage",
       y = "Topic Probability") +
  theme_minimal()

# TODO:
data_set_with_topics_and_metadata<- data_set_with_topics_and_metadata %>%group_by(ticker) %>%summarise_all(mean(topics)) 
#Using PCA, plot the first two coordinates in a scatter (dim1,dim2) and color code by company type 

# subset to fossil fuel companies
lm(financial_1~topic1) 
m(financial_1~topic1)
m(financial_1~topick)
lm(financial_1~topic1+..+topic_k)
lm(financial_2~topic1+..+topic_k)

