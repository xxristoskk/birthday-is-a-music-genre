import pickle
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

##functions

### code i found on stackoverflow
def scroll_to_bottom(driver):
    SCROLL_PAUSE_TIME = 2.5

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

### grabs all the genre tags from the tags index
def get_tags():
    driver = webdriver.Firefox()
    driver.get('https://bandcamp.com/tags')
    time.sleep(2.5)
    driver.find_elements_by_class_name('showall')[0].click()
    time.sleep(2.5)
    cloud = driver.find_elements_by_class_name('tagcloud')[0]
    tag_links = cloud.find_elements_by_tag_name('a')
    tags = [link.text for link in tag_links]
    return tags

### removes any empty strings from the scraper function
def remove_empty_strings(titles):
    txt = []
    for title in titles:
        if len(title.text) < 1:
            continue
        else:
            txt.append(title.text)
    return txt

## scrapes all the releases from each genre webpage
def super_scraper(tags):
    case = []
    driver = webdriver.Firefox()
    for tag in tqdm(tags):
        tag_dict = {}
        driver.get(f'https://bandcamp.com/tag/{tag}?tab=all_releases')
        time.sleep(1.7)
        view_more = driver.find_elements_by_class_name('view-more')[0]
        view_more.click()
        time.sleep(1.7)
        scroll_to_bottom(driver)
        title_txt = remove_empty_strings(driver.find_elements_by_class_name('title'))
        artist_txt = remove_empty_strings(driver.find_elements_by_class_name('artist'))
        artist_txt = remove_by(artist_txt)
        tag_dict[tag] = {x[0]:x[1] for x in list(zip(artist_txt,title_txt))}
        case.append(tag_dict)
        pickle.dump(case,open('bc_genre_releases.pickle','wb'))
        case = pickle.load(open('bc_genre_releases.pickle','rb'))
        time.sleep(1)
    return case
## removes "by" Artist
def remove_by(lis):
    case = []
    for item in lis:
        case.append(item.split('by ')[1])
    return case

## remove duplicate items
def remove_duplicates(lis):
    l1=[]
    l2=[]
    for item in lis:
        if item not in l1:
            l1.append(item)
        else:
            l2.append(item)
    return l2

tags = get_tags()
for i,tag in enumerate(tags):
    if '/' in tag:
        tags[i] = tags[i].replace('/','-')

super_scraper(tags)
