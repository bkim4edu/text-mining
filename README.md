## Project Overview
I used IMDB as a data source to grab reviews of some top movies as rated by IMDB. I ran these reviews through NLTK’s sentiment analysis to try and determine whether the review was positive or negative. Next, I used Project Gutenberg to grab some works by Charles Dickens to then run a Markov Text Synthesis to create three new sentences. I also ran statistics on the most common words found in the first 50,000 words of the selected Dickens works. I wanted to use this assignment as an opportunity to become more comfortable with recent material such as lists and I feel like I did make decent progress in this regard.

## Implementation
**IMDB Movie Reviews**

I started by creating a loop that pulls the listed movies and their reviews, cleaning and processing them to all be consistent in terms of capitalization and removing common words with no significant contribution to the meaning of the review. I then ran NLTK’s sentiment intensity analysis on the reviews. If the positive score is greater than the negative score, the review is deemed positive; otherwise it is deemed negative. The next step would have been to separate the reviews and cumulate the scores of the reviews to print one statement of whether it was positive or negative. However, I committed too much time to finishing other components of the assignment thinking this step would be simple to complete. It was not and I wasn’t able to completely finish this portion.

**Charles Dickens Analyses**

I found three Dickens works on Project Gutenberg to analyze. I cleaned the text and limited the analyses to the first 50,000 words. I used all three works as inputs for Markovify to synthesize three new sentences. Next was a summary statistic on the ten most common words from each work. For this, I first further cleaned the text removing all punctuation before outputting the ten most common words and visualizing it on a graph using matplotlib.
 
## Results
It was interesting to look at the most common words specific to each book and relate them back to the plots and storylines of the works. The Markov text synthesis was the most interesting part of this for me. Natural language models are a fascination for me, especially models such as GPT-3. Although most fun parts of the logic were handled behind the scenes by the Markovify library, it was still fun to be able to curate my own inputs that the Markov text syntheses would be based off of.
The positive/negative review analyzer was what first came to mind when brainstorming what to include in this assignment and what I was most excited about initially. In my opinion, the biggest accomplishment of this was writing code to pull data from IMDB and adapting that to be analyzed by NLTK, as again most of the legwork of deciding the positive/negative sentiment of the text was handled behind the scenes by NLTK.

Sample Markov Text Synthesis: "All this time, Sir Matthew Pupker and the two other real members of Parliament are positively coming. A lady in deep mourning rose as Mr. Ralph Nickleby shrugged his shoulders, and said things were better as they were. There is that in the river too."

![img1](https://i.imgur.com/2SyAyA2.png)
![img2](https://i.imgur.com/W1yVeyD.png)
![img3](https://i.imgur.com/is4SKf4.png)

## Reflection
Overall, I’m satisfied with my work on this assignment. I felt I was falling behind on the more recent material of lists, dictionaries, tuples, etc. Although I’m still not confident on the material, I felt like this assignment has helped a lot in catching back up to some extent. I feel like my assignment was appropriately scoped, but that I just didn’t give myself enough time to work on it. Especially with the IMDB positive/negative review portion, I made enough progress early on that I was confident it would be the easy portion of my assignment and moved on to working on other portions of the assignment. With this write-up portion too, I wish I had left myself enough time to go more in-depth rather than scramble to get something together before the deadline. 
