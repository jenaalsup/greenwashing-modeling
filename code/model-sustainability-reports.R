#### Load House Data 

textData <- read.csv("reports/processed-data.csv")
textData <- textData[!duplicated(textData$doc_id),]

# create a week
textData$date <- as.POSIXct(
  paste0(textData$year,"-",textData$month,"-",textData$day),
  format="%Y-%b-%e")
textData$week   <- isoweek(textData$date)

## prepare JST model
attach(textData)
llDisplay <- data.frame(doc_id = paste(doc_id),text = paste(text),week=week,handle=handle,month=month,day=day,year=year,party=party,state=state,stringsAsFactors = F)
detach(textData)

stop_words_custom = c("will","has","by","for","hi","hey","are","as","i","we","our","ours","ourselves","use",
                       "you","your","yours","that","f","e","s","t","c","n","u","v","l","p","d","b","g","k","m","x","y","z",
                       "be","with","is","was","been","not","they","way","and","to","do","go","on","have","from",
                       "at","but","or","an","if","all","so","it","thing","put","well","take","see","","can't","can",
                       "got","cant","could","him","his","this","had","he","her","she","hers","their","they're","things",
                       "go","going","let","would","make","like","come","us",paste(textData$firstname),paste(tidytext::stop_words$word),stopwords())
# Create tokens for the words you keep
toks1 <- tokens(corpus(llDisplay), remove_punct = TRUE)
toks2 <- tokens_remove(toks1, stop_words_custom)
toks3 <- tokens_ngrams(toks2, 1:3)
toks3 <- tokens_wordstem(toks3)
#toks3 <- tokens_keep(toks3,dfm_speeches@Dimnames$features)


N = nrow(llDisplay)

# create a document feature matrix (quanteda style) to run JST
dfm_speeches <- dfm(toks3 , 
                    remove_punct = TRUE ,remove_numbers = TRUE,stem = T) %>% 
  dfm_trim(min_termfreq = 25,min_docfreq =0.002*N,max_docfreq = 0.4*N)




dfm_speeches <- dfm_subset(dfm_speeches,ntoken(dfm_speeches) >= 1)

## Prep data for STM
houseDfmSTM <- convert(dfm_speeches, to = "stm", 
                       docvars = docvars(dfm_speeches))

houseDfmSTM$meta$date <- as.POSIXct(
  paste0(houseDfmSTM$meta$year,"-",houseDfmSTM$meta$month,"-",houseDfmSTM$meta$day),
  format="%Y-%b-%e")

documents <- houseDfmSTM$documents
vocab     <- houseDfmSTM$vocab
meta      <- houseDfmSTM$meta


# select number of topics
kk = c(5,10,15,20,25,28,30,40,50,60)  
print("Run STM")
model_stm <- mclapply(kk,
                      FUN = function(kk) 
                        stm(houseDfmSTM$documents, 
                            houseDfmSTM$vocab,
                            K = kk,
                            prevalence=~factor(party)  + factor(state) + as_date(date),
                            max.em.its = 200,
                            data = houseDfmSTM$meta, 
                            init.type = "Spectral",
                            seed = 153332),
                      mc.cores = getOption("mc.cores", 
                                           5L)) 

save(model_stm,file="modelOutput-House-STM.RData")
