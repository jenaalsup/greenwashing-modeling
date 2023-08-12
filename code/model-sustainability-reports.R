#### Load House Data 

textData <- read.csv("Data/TwitterSpeech.csv")
textData <- textData[!duplicated(textData$doc_id),]

# create a week
textData$date <- as.POSIXct(
  paste0(textData$year,"-",textData$month,"-",textData$day),
  format="%Y-%b-%e")
textData$week   <- isoweek(textData$date)

## prepare JST model
attach(textData)
llDisplay <- data.frame(doc_id = paste(doc_id),text = paste(tweet),week=week,handle=handle,month=month,day=day,year=year,party=party,state=state,stringsAsFactors = F)
detach(textData)

stop_words_custom = c("this","discuss","community","learn","congressional","amendment","speaker","say","said","talk","congrats","pelosi","gop","congratulations","are","as","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours","he","her","him","she","hers","that","be","with","their","they're","is","was","been","not","they","it","have","will","has","by","for","madam","Speaker","Mister","Gentleman","Gentlewoman","lady","voinovich","kayla","111th","115th","114th","rodgers",         "clerk" ,    "honor" ,   "address"   ,     
                      "house" , "start"   ,"amend","bipartisan","bill",   "114th"    ,   "congress"  ,     
                      "one",   "thing"    ,   "learned"    ,   "legislator","things" ,"things","can't","can","cant","will","go","going","let","lets","let's","say","says","know","talk","talked","talks","lady","honorable","dont","think","said","something","something","wont","people","make","want","went","goes","congressmen","people","person","like","come","from","need","us",paste(textData$firstname),paste(tidytext::stop_words$word),stopwords())
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
