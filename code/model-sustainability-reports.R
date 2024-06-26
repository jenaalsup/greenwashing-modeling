# load libraries
library(tidytext)
library(dplyr)
library(tidyverse)
library(quanteda)
library(parallel)
library(stm)
library(tm)
library(ggplot2)
library(wordcloud)
library(RColorBrewer)
library(stats)
library(factoextra)
library(viridis)
library(stringr)
library(tidyr)

# load data
textData <- read.csv("reports/processed_reports/processed-data.csv")
textData <- textData[!duplicated(textData$doc_id),]
capex_data <- read_csv("reports/processed_reports/capex-data.csv")
textData <- textData %>%left_join(capex_data)

# prepare model
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

# create a document feature matrix (quanteda style) to run model
N = nrow(llDisplay)
dfm_speeches <- dfm(toks3, remove_punct = TRUE, remove_numbers = TRUE, stem = T) %>% 
  dfm_trim(min_termfreq = 15, min_docfreq =0.01*N, max_docfreq = 0.8*N) 
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

save(model_stm,file="modelOutput-House-STM.RData")

# view topic stats
labelTopics(model_stm[[4]])

# calculate umass coherence
coherence_scores <- lapply(model_stm, function(model) {
  semanticCoherence(model, documents)
})
average_coherence_per_model <- sapply(coherence_scores, mean)
print(average_coherence_per_model)

# prepare semantic coherence score data for plotting
coherence_scores_plot <- data.frame(
  Number_of_Topics = kk,
  Avg_Coherence_Score = average_coherence_per_model
)

# plot average semantic coherence scores vs. number of topics
ggplot(coherence_scores_plot, aes(x = Number_of_Topics, y = Avg_Coherence_Score)) +
  geom_point() +
  geom_line() +  # Add a line connecting the points
  labs(
    x = "Number of Topics",
    y = "Semantic Coherence Score",
    title = "Semantic Coherence by Number of Topics"
  )

# topic selection: 4th model, 20 topics, topics 5 & 14
model_stm[[4]]$theta # doc-topic matrix - row is doc and column is probability it is in that topic
model_stm[[4]]$theta[, 5] # wildlife/conservation/biodiversity topic
model_stm[[4]]$theta[, 14] # climate/carbon topic

# generate word clouds
topic_5_gradient_colors <- colorRampPalette(c("darkblue", "blue", "deepskyblue", "darkturquoise", "cadetblue3"))(20)
cloud(model_stm[[4]], topic = 5, scale = c(2,.25), max.words = 40, color = topic_5_gradient_colors)
topic_14_gradient_colors <- colorRampPalette(c("darkgreen", "forestgreen", "chartreuse3", "darkseagreen3", "darkseagreen2"))(20)
cloud(model_stm[[4]], topic = 18, scale = c(2,.25), max.words = 40, color = topic_14_gradient_colors)

# plot env capex percentage data for energy firms
filtered_data <- textData %>%
  filter(company_type == "energy")
agg_data <- filtered_data %>%
  group_by(company_ticker) %>%
  summarise(total_spending_percentage = 100 * sum(coalesce(env_capex, 0)) / sum(coalesce(capex, 0)))
color_palette <- scales::viridis_pal(option = "D")(length(unique(agg_data$company_ticker)))
agg_data$total_spending_percentage[is.na(agg_data$total_spending_percentage)] <- 0
ggplot(agg_data, aes(x = reorder(str_to_upper(company_ticker), total_spending_percentage), y = total_spending_percentage, fill = total_spending_percentage)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "firebrick1", high = "chartreuse3", guide = "legend") +
  labs(x = "Company Ticker Symbol", y = "Environmental Percentage of Capex", title = "Environmental Spending by Company") +
  scale_y_continuous(limits = c(0, 18))

# generate a principal component analysis (pca)
pca_data <- model_stm[[4]]$theta
pca_data <- cbind(textData,pca_data)
pca_input <-pca_data    %>% 
  group_by(company_ticker,company_type)  %>%
  select(-capex,-env_capex,-year,-doc_id,-text,-part) %>%
  summarise_all(mean)
pca_mat<-as.matrix(pca_input%>%ungroup()%>%select(-company_ticker,-company_type))

fit  <-prcomp(pca_mat)
res.var <- get_pca_var(fit) # output which variables (topics) contribute the most to each dimension ~ aka the variation in the data
ind.var <- get_pca_ind(fit)
pc1 <- ind.var$coord[,1]
pc2 <- ind.var$coord[,2]
pca_input$dim1 <- pc1 # first principle component
pca_input$dim2 <- pc2 # second principle component
res.var$contrib # see dim topic contributions

ggplot(pca_input,aes(x=dim1,y=dim2,colour=company_type))+geom_point() +
  labs(title = "Principal Component Analysis",
       x = "Solar/Green Topic",
       y = "Electricity/Power/Efficiency Topic",
       color = "Company Type") +
  scale_color_manual(
    guide = guide_legend(),
    labels = c('Clean Energy', 'Energy'),
    values = c('springgreen3', 'firebrick2')
  )

# correlate the green columns of theta (the topics) with the capex data
averaged_data <- model_stm[[4]]$theta
averaged_data <- cbind(textData,averaged_data)
averaged_data <-averaged_data %>% group_by(company_ticker,year,company_type) %>% 
                                  summarise(capex = mean(env_capex/capex),
                                  topic_5  = mean(`5`),
                                  topic_14 = mean(`14`)) %>%
                                  mutate(topic_avg=(topic_5+topic_14)/2)
# don't group by year:
#averaged_data_company <-averaged_data %>% group_by(company_ticker,company_type) %>% 
                                  #summarise(capex = mean(env_capex/capex),
                                  #topic_5  = mean(`5`),
                                  #topic_14 = mean(`14`)) %>%
                                  #mutate(topic_avg=(topic_5+topic_14)/2)

# plots of stm results vs. capex data (6):

# plot topic probability (avg of topics 5 and 14) vs. env capex percentage (each dot is a document)
ggplot(averaged_data %>%filter(company_type=="energy"), aes(x = 100*capex, y = topic_avg)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color="darkorchid3") + # method = "lm" adds regression, se = TRUE adds confidence interval
  labs(title = "Topic Probability vs. Env Capex Percentage",
       x = "Environmental Capex Percentage",
       y = "Environmental Topic Probability (avg)") +
  theme_minimal()

# plot topic 5 probability vs. env capex percentage
ggplot(averaged_data %>%filter(company_type=="energy"), aes(x = 100*capex, y = topic_5)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color="firebrick2") + # method = "lm" adds regression, se = TRUE adds confidence interval
  labs(title = "Topic Probability vs. Env Capex Percentage",
       x = "Environmental Capex Percentage",
       y = "Conservation/Biodiversity Topic Probability") +
  theme_minimal()

# plot topic 14 probability vs. env capex percentage
ggplot(averaged_data %>%filter(company_type=="energy"), aes(x = 100*capex, y = topic_14)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color="royalblue2") + # method = "lm" adds regression, se = TRUE adds confidence interval
  labs(title = "Topic Probability vs. Env Capex Percentage",
       x = "Environmental Capex Percentage",
       y = "Climate Topic Probability") +
  theme_minimal()

# plot topic probability (avg of topics 5 and 14) vs. env capex percentage (each dot is a document) for documents >2020
ggplot(averaged_data %>%filter(company_type=="energy",year>2020), aes(x = 100*capex, y = topic_avg)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color="darkorchid3") + # method = "lm" adds regression, se = TRUE adds confidence interval
  labs(title = "Topic Prob. vs. Env Capex (years > 2020)",
       x = "Environmental Capex Percentage",
       y = "Environmental Topic Probability (avg)") +
  theme_minimal()

# plot topic 5 probability vs. env capex percentage for documents >2020
ggplot(averaged_data %>%filter(company_type=="energy",year>2020), aes(x = 100*capex, y = topic_5)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color="firebrick2") + # method = "lm" adds regression, se = TRUE adds confidence interval
  labs(title = "Topic Prob. vs. Env Capex (years > 2020)",
       x = "Environmental Capex Percentage",
       y = "Conservation/Biodiversity Topic Probability") +
  theme_minimal()

# plot topic 14 probability vs. env capex percentage for documents >2020
ggplot(averaged_data %>%filter(company_type=="energy",year>2020), aes(x = 100*capex, y = topic_14)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE, color="royalblue2") + # method = "lm" adds regression, se = TRUE adds confidence interval
  labs(title = "Topic Prob. vs. Env Capex (years > 2020)",
       x = "Environmental Capex Percentage",
       y = "Climate Topic Probability") +
  theme_minimal()


###############
# 2024 updates
###############

# wordclouds over time
textData_before_2018 <- subset(textData, year < 2018)
textData_after_2018 <- subset(textData, year >= 2018)
N_before_2018 <- nrow(textData_before_2018)
dfm_speeches_before_2018 <- dfm(tokens(corpus(textData_before_2018$text), remove_punct = TRUE, remove_numbers = TRUE, stem = TRUE)) %>%
  dfm_trim(min_termfreq = 15, min_docfreq = 0.01 * N_before_2018, max_docfreq = 0.8 * N_before_2018)
N_after_2018 <- nrow(textData_after_2018)
dfm_speeches_after_2018 <- dfm(tokens(corpus(textData_after_2018$text), remove_punct = TRUE, remove_numbers = TRUE, stem = TRUE)) %>%
  dfm_trim(min_termfreq = 15, min_docfreq = 0.01 * N_after_2018, max_docfreq = 0.8 * N_after_2018)
model_stm_before_2018 <- stm(documents = dfm_speeches_before_2018, 
                             prevalence = ~ factor(company_type) + factor(year),
                             K = 20, max.em.its = 200,
                             data = textData_before_2018,
                             init.type = "Spectral", seed = 153332)
model_stm_after_2018 <- stm(documents = dfm_speeches_after_2018, 
                            prevalence = ~ factor(company_type) + factor(year),
                            K = 20, max.em.its = 200,
                            data = textData_after_2018,
                            init.type = "Spectral", seed = 153332)
cloud(model_stm_before_2018, topic = 5, scale = c(2, .25), max.words = 40, color = topic_5_gradient_colors)
cloud(model_stm_after_2018, topic = 5, scale = c(2, .25), max.words = 40, color = topic_5_gradient_colors)
cloud(model_stm_before_2018, topic = 14, scale = c(2, .25), max.words = 40, color = topic_14_gradient_colors)
cloud(model_stm_after_2018, topic = 14, scale = c(2, .25), max.words = 40, color = topic_14_gradient_colors)

# wordclouds by firm type
textData_clean_energy <- subset(textData, company_type == "clean-energy")
textData_energy <- subset(textData, company_type == "energy")
N_clean_energy <- nrow(textData_clean_energy)
dfm_speeches_clean_energy <- dfm(tokens(corpus(textData_clean_energy$text), remove_punct = TRUE, remove_numbers = TRUE, stem = TRUE)) %>%
  dfm_trim(min_termfreq = 15, min_docfreq = 0.01 * N_clean_energy, max_docfreq = 0.8 * N_clean_energy)
N_energy <- nrow(textData_energy)
dfm_speeches_energy <- dfm(tokens(corpus(textData_energy$text), remove_punct = TRUE, remove_numbers = TRUE, stem = TRUE)) %>%
  dfm_trim(min_termfreq = 15, min_docfreq = 0.01 * N_energy, max_docfreq = 0.8 * N_energy)
model_stm_clean_energy <- stm(documents = dfm_speeches_clean_energy, 
                             prevalence = ~ factor(year),
                             K = 20, max.em.its = 200,
                             data = textData_clean_energy,
                             init.type = "Spectral", seed = 153332)
model_stm_energy <- stm(documents = dfm_speeches_energy, 
                            prevalence = ~ factor(year),
                            K = 20, max.em.its = 200,
                            data = textData_energy,
                            init.type = "Spectral", seed = 153332)
cloud(model_stm_clean_energy, topic = 5, scale = c(2, .25), max.words = 40, color = topic_5_gradient_colors)
cloud(model_stm_energy, topic = 5, scale = c(2, .25), max.words = 40, color = topic_5_gradient_colors)
cloud(model_stm_clean_energy, topic = 14, scale = c(2, .25), max.words = 40, color = topic_14_gradient_colors)
cloud(model_stm_energy, topic = 14, scale = c(2, .25), max.words = 40, color = topic_14_gradient_colors)


