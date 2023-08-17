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


labelTopics(model_stm[[1]])

library(stm)

coherence_scores <- lapply(model_stm, function(model) {
  semanticCoherence(model, documents)
})

# Calculate the average coherence score for each model
average_coherence_per_model <- sapply(coherence_scores, mean)

# Print the average coherence scores
print(average_coherence_per_model)

### Google how to compute perplexity and coherence, topic selection 

# After topic selection
model_stm[[1]]$theta # doc-topic matrix
model_stm[[1]]$settings$covariates # covariates 

## Once you're happy with the model and did topic selection, correlate the columns of theta(ie the topics) which are green with the financial data
## Plos the scatter with the slope

## Use ggplot scatter with regression slope (confidence interval)

data_set_with_topics_and_metadata<- data_set_with_topics_and_metadata %>%group_by(ticker) %>%summarise_all(mean(topics)) 
#Using PCA, plot the first two coordinates in a scatter (dim1,dim2) and color code by company type 

# subset to fossil fuel companies
lm(financial_1~topic1) 
m(financial_1~topic1)
m(financial_1~topick)
lm(financial_1~topic1+..+topic_k)
lm(financial_2~topic1+..+topic_k)


