import os.path

import newspaper as N
import numpy as np
import pandas as pd
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from scipy.stats import norm, zscore

SITES = {
    'CNN': 'http://www.cnn.com/',
    'Fox': 'http://www.foxnews.com/',
    'NPR': 'https://www.npr.org/sections/news/',
    'BBC': 'http://www.bbc.com/news/world/us_and_canada',
    'NYT': 'https://www.nytimes.com/',
    'WaPo': 'https://www.washingtonpost.com/',
    'BNN': 'http://www.breitbart.com/',
    'Gdn': 'https://www.theguardian.com/us',
    'Pol': 'https://www.politico.com/',
    'ABC': 'http://abcnews.go.com/',
    'Huff': 'https://www.huffingtonpost.com/'
}

TRUMPISMS = [
    'donald j. trump',
    'donald j trump',
    'donald trump',
    'president',
    'trump',
    'donald',
    'administration'
]


def get_trump_score(url):
    '''
    Gets the average sentiment values for all Trump articles at the given url

    :param url:
        The website to crawl for articles
    :return:
        The entity sentiment analysis score and magnitude. See Google cloud api documenation for more info
    '''
    # Get article text
    site = N.build(url, memoize_articles=False)

    scores = []
    for art in site.articles:
        if 'politics' in art.url or 'trump' in art.url.lower():
            art.download()
            art.parse()
            if 'trump' in art.title.lower() or 'trump' in art.text.lower():
                scores.append(get_article_score(art.text))
    filtered_scores = list(filter(lambda s: s is not None, scores))
    if len(filtered_scores) == 0:
        return None
    sc, mag = np.array(filtered_scores).T
    return np.average(sc), np.average(mag)


def get_article_score(text):
    '''
    Gets the average entity sentiment value for all trumpisms in the given text

    :param text:
        A long string to analyze using Google cloud api entity sentiment analysis
    :return:
        An entity score (from -1 to 1) and magnitude. -1 is most negative and 1 is most positive
    '''
    # Instantiates a client
    client = language.LanguageServiceClient()
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)
    try:
        result = client.analyze_entity_sentiment(document=document)
    except:
        return None
    sents = []
    for entity in result.entities:
        if entity.name.lower() in TRUMPISMS:
            sent = entity.sentiment
            if sent.score != 0 or sent.magnitude != 0:
                sents.append((sent.score, sent.magnitude))
    if len(sents) == 0:
        return None
    score, mag = np.array(sents).T
    return np.average(score), np.average(mag)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from seaborn import distplot, barplot

    if os.path.isfile('site_data.csv'):
        print('Data file found')
        data = pd.read_csv('data/site_data.csv')
    else:
        print('Caluclating new data')
        output = list(map(get_trump_score, SITES.values()))
        mask = list(map(lambda s: s is not None, output))
        scores, mags = list(zip(*np.array(output)[mask]))
        sites = np.array(list(SITES))[mask]
        data = pd.DataFrame()
        data['Score'] = scores
        data['Mag'] = mags
        data['Site'] = sites
        data['z-value'] = zscore(scores)
        low, upp = norm.interval(.95)
        data['signif'] = list(map(lambda z: z < low or z > upp, scores))
        data.to_csv('data/site_data.csv')

    for site, score, sig in zip(data['Site'], data['Score'], data['signif']):
        if sig: print('{} is significantly different in sentiment with a score of {}'.format(site, score))

    barplot(data['Site'], data['Score'], color='C0')
    plt.show()
    distplot(data['Score'], kde=False, fit=norm)
    plt.title('u={:.3f} s={:.4f}'.format(*norm.fit(data['Score'])))
    plt.show()
