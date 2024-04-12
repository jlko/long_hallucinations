"""FactualBio dataset for paragraph-length hallucination detection."""

# pylint: disable=line-too-long


# Documentation
# -------------
# * Column format: Input, Original Output, Truth (Sentence-Level),
#       Extracted Facts, Truth (Fact-Level)
# * Difference between MINOR and MAJOR errors: MINOR false are small mistakes,
#       e.g. a year or a place that is wrong. Sometimes when going from
#       sentences to facts, the MINOR mistake will appear more prominently,
#       e.g. 'His Name is, he did this and that, and he was born in 1968'
#       becomes 'He was born in 1968'. In these cases, we label the
#       mistake as MINOR on a sentence- and fact-level.
#       We currently default to treating both MINOR and MAJOR errors as errors.
# * Fact extraction: We have used the following prompt with GPT-4 to extract
#   the facts from the sentences:
#   'Please list the specific factual propositions included in the answer above. Be complete and do not leave any factual claims out. Provide each claim as a separate sentence in a separate bullet point.'


MAJOR = 'Major False'
MINOR = 'Minor False'


data = [
    [
        0,
        'Who is John Russel Reynolds?',
        'Sir John Russell Reynolds was a notable British physician who was born in 1828 and died in 1896. He was a highly reputed neurologist, and served as physician in ordinary to Queen Victoria for many years. Reynolds was known for his research on epilepsy, for which he is still recognized. He was president of the Royal College of Physicians from 1881 to 1888 and was knighted in 1893. He also held the prestigious position of professor of clinical medicine at University College Hospital, London.',
        [True, True, True, MAJOR, True],
        [
            'Sir John Russell Reynolds was a British physician.',
            'He was born in 1828.',
            'He died in 1896.',
            'Reynolds was a notable neurologist.',
            'He served as physician in ordinary to Queen Victoria for many years.',
            'Reynolds was known for his research on epilepsy.',
            'He was president of the Royal College of Physicians from 1881 to 1888.',
            'He was knighted in 1893.',
            'Reynolds held the position of professor of clinical medicine at University College Hospital, London.',
        ],
        [True, True, True, True, True, True, MINOR, MAJOR, True]
    ],
    [
        1,
        'Who is Adja Yunkers?',
        'Adja Yunkers was a Latvian-American abstract painter, printmaker, and book illustrator. He was born in Riga, Latvia, in 1900 and died in New York City, USA in 1983. Yunkers studied art in Leningrad, Berlin, Paris, and London and traveled extensively before settling down in the United States in 1947. His work is characterized by its inventive style, incorporating elements of Abstract Expressionism.',
        [True, True, True, True],
        [
            'Adja Yunkers was a Latvian-American abstract painter, printmaker, and book illustrator.',
            'Yunkers was born in Riga, Latvia, in 1900.',
            'Yunkers died in New York City, USA in 1983.',
            'Yunkers studied art in Leningrad, Berlin, Paris, and London.',
            'Yunkers traveled extensively before settling down in the United States in 1947.',
            'His work is characterized by its inventive style.',
            'His work incorporates elements of Abstract Expressionism.',
        ],
        [True, True, True, True, True, True, True],
    ],
    [
        2,
        'Who is Bert Deacon?',
        "Bert Deacon was a former Australian rules footballer who played for Carlton Football Club in the Victorian Football League (VFL). He is widely considered one of the greatest players of his era, having won the Brownlow Medal in 1947, which is awarded to the 'best and fairest' player in the league. Deacon was the first Carlton player to win this prestigious award. He was born on 29 November 1922 and passed away on 5 January 1984. Deacon was inducted into the Australian Football Hall of Fame in 1996.",
        [True, True, True, MINOR, MAJOR],
        [
            'Bert Deacon was an Australian rules footballer.',
            'He played for Carlton Football Club in the Victorian Football League (VFL).',
            'He is widely considered one of the greatest players of his era.',
            'Deacon won the Brownlow Medal in 1947.',
            'The Brownlow Medal is awarded to the \'best and fairest\' player in the league.',
            'Deacon was the first Carlton player to win the Brownlow Medal.',
            'He was born on 29 November 1922.',
            'He passed away on 5 January 1984.',
            'Deacon was inducted into the Australian Football Hall of Fame in 1996.',
        ],
        [True, True, True, True, True, True, MINOR, MINOR, MAJOR],
    ],
    [
        3,
        "Who is Freddie Frith?",
        'Freddie Frith was an English motorcycle road racer who became a champion in both pre-World War II and post-war eras. He won the 1935 and 1937 Grand Prix motorcycle racing European Championships. After retiring from competition, he became the president of the Auto Cycle Union, the governing body of British motorcycle racing. He was also an accomplished motorcycle dealer and manufacturer. Frith was born in 1911 and died in 1988.',
        [True, MINOR, MAJOR, MAJOR, MINOR],
        [
            'Freddie Frith was an English motorcycle road racer.',
            'He won the 1935 and 1937 Grand Prix motorcycle racing European Championships.',
            'After retiring from competition, he became the president of the Auto Cycle Union.',
            'The Auto Cycle Union is the governing body of British motorcycle racing.',
            'Frith was also an accomplished motorcycle dealer and manufacturer.',
            'Frith was born in 1911.',
            'Frith died in 1988.',
        ],
        [True, MINOR, MAJOR, True, MAJOR, MINOR, True],
    ],
    [
        4,
        'Who is Wilhelm Windelband?',
        'Wilhelm Windelband was a German philosopher who lived from 1848 to 1915. He is considered one of the key figures in the development of neo-Kantianism, a school of thought that revisited the ideas put forth by Immanuel Kant during the 18th century. Windelband was particularly focused on history and culture and is noted for his concept of "idiographic" and "nomothetic" approaches to study these fields. He served as a professor at several universities throughout his career, including the University of Berlin and the University of Heidelberg.',
        [True, True, True, MINOR],
        [
            'Wilhelm Windelband was a German philosopher.',
            'He lived from 1848 to 1915.',
            'He is considered one of the key figures in the development of neo-Kantianism.',
            'Neo-Kantianism is a school of thought that revisited the ideas put forth by Immanuel Kant during the 18th century.',
            'Windelband was particularly focused on history and culture.',
            'He is noted for his concept of "idiographic" and "nomothetic" approaches to study history and culture.',
            'He served as a professor at several universities.',
            'These universities include the University of Berlin and the University of Heidelberg.',
        ],
        [True, True, True, True, True, True, True, MINOR],
    ],
    [
        5,
        "Who is Moisés Kaufman?",
        'Moisés Kaufman is a Venezuelan playwright and director, known for his work in the theater community. He is the founder of the Tectonic Theater Project and has won several awards for his plays and directing, including an Emmy for his film adaptation of "The Laramie Project". Some of his other notable works include "Gross Indecency: The Three Trials of Oscar Wilde" and "33 Variations". Kaufman is also an advocate for LGBTQ+ rights and diversity in the arts.',
        [True, MINOR, True, True],
        [
            'Moisés Kaufman is a Venezuelan playwright and director.',
            'He is known for his work in the theater community.',
            'He is the founder of the Tectonic Theater Project.',
            'He has won several awards for his plays and directing.',
            'He won an Emmy for his film adaptation of "The Laramie Project".',
            'Some of his other notable works include "Gross Indecency: The Three Trials of Oscar Wilde" and "33 Variations".',
            'Kaufman is an advocate for LGBTQ+ rights.',
            'He also advocates for diversity in the arts.',
        ],
        [True, True, True, True, MINOR, True, True, True],
    ],
    [
        6,
        'Who is Laurent Koscielny?',
        "Laurent Koscielny is a retired professional footballer who is from France. He spent the majority of his career playing as a central defender for Arsenal Football Club in England's Premier League. He also played for the French national team. Koscielny was known for his technical skills, intelligence, and athleticism. He retired from professional football in 2021, and he now works as a coach for Bordeaux, a club in France’s Ligue 1.",
        [True, True, True, True, MINOR],
        [
            'Laurent Koscielny is a retired professional footballer from France.',
            'He spent the majority of his career playing as a central defender for Arsenal Football Club in England\'s Premier League.',
            'He also played for the French national team.',
            'Koscielny was known for his technical skills, intelligence, and athleticism.',
            'He retired from professional football in 2021.',
            'He now works as a coach for Bordeaux, a club in France’s Ligue 1.',
        ],
        [True, True, True, True, MINOR, MINOR],
    ],
    [
        7,
        "Who is Véra Korène?",
        'Véra Korène was a Russian-born French actress and singer, known for her roles in French cinema during the 1930s. She was born on December 10, 1901 in Saint Petersburg, Russia and died on November 15, 1996 in Paris, France. Some of her notable films include "La nuit est à nous" (1930), "Un carnet de bal" (1937), and "Le messager" (1937).',
        [True, MINOR, MAJOR],
        [
            'Véra Korène was a Russian-born French actress and singer.',
            'She was known for her roles in French cinema during the 1930s.',
            'She was born on December 10, 1901 in Saint Petersburg, Russia.',
            'She died on November 15, 1996 in Paris, France.',
            'Some of her notable films include "La nuit est à nous" (1930), "Un carnet de bal" (1937), and "Le messager" (1937).',
        ],
        [True, True, MINOR, MINOR, MAJOR, ],
    ],
    [
        8,
        'Who is Leana de Bruin?',
        'Leana de Bruin is a professional netball player from South Africa who later represented New Zealand. She was born on 9 July 1977. She is considered one of the best goalkeepers in the sport and has won numerous accolities throughout her career, including World Netball Championships, Commonwealth Games and ANZ Championship titles. She retired from international netball in 2017.',
        [True, True, MINOR, MAJOR],
        [
            'Leana de Bruin is a professional netball player.',
            'She is from South Africa.',
            'She later represented New Zealand.',
            'She was born on 9 July 1977.',
            'She is considered one of the best goalkeepers in netball.',
            'She has won World Netball Championships, Commonwealth Games and ANZ Championship titles.',
            'She retired from international netball in 2017.'
        ],
        [True, True, True, True, MINOR, MINOR, MAJOR],
    ],
    [
        9,
        'Who is Tera Van Beilen?',
        'Tera Van Beilen is a Canadian competitive swimmer who specializes in breaststroke events. She represented Canada at the 2012 Summer Olympics. Van Beilen was born on February 19, 1993, in Oakville, Ontario, Canada. She went on to compete for the University of British Columbia.',
        [True, True, MINOR, True],
        [
            'Tera Van Beilen is a Canadian competitive swimmer.',
            'She specializes in breaststroke events.',
            'She represented Canada at the 2012 Summer Olympics.',
            'Van Beilen was born on February 19, 1993.',
            'She was born in Oakville, Ontario, Canada.',
            'She competed for the University of British Columbia.',
        ],
        [True, True, True, MINOR, MINOR, True]
    ],
    [
        10,
        'Who is Lucy Akhurst?',
        'Lucy Akhurst is an English actress, director, and writer. She has appeared in a number of films, including "All the Little Animals," "Trinity," and "The Climb." Beyond her acting career, Akhurst has also gained acclaim for her writing. She co-wrote the screenplay for "The Man Who Knew Too Little." Her success in a variety of roles in the entertainment industry has made her a notable figure in the world of British film and television.',
        [True, 'MAJOR', 'MAJOR'],
        [
            'Lucy Akhurst is an English actress, director, and writer.',
            'She has appeared in a number of films, including "All the Little Animals," "Trinity," and "The Climb."',
            'Akhurst has also gained acclaim for her writing.',
            'She co-wrote the screenplay for "The Man Who Knew Too Little."',
            'Her success in a variety of roles in the entertainment industry has made her a notable figure in the world of British film and television.'
        ],
        [True, MAJOR, True, MAJOR, True]
    ],
    [
        11,
        'Who is Alana Davis?',
        'Alana Davis is an American singer-songwriter. Her biggest hit to date is a cover of the Ani DiFranco song "32 Flavors" which reached the Billboard Top 40. Her music is typically classified as folk rock and pop. She first came into prominence in the mid to late 1990s and has continued to produce music since then, though she has not repeated her initial chart success.',
        [True, True, True, True],
        [
            'Alana Davis is an American singer-songwriter.',
            'Her biggest hit to date is a cover of the Ani DiFranco song "32 Flavors."',
            '"32 Flavors" reached the Billboard Top 40.',
            'Her music is typically classified as folk rock and pop.',
            'She first came into prominence in the mid to late 1990s.',
            'She has continued to produce music since then.',
            'She has not repeated her initial chart success.'],
        [True, True, True, True, True, True, True]
    ],
    [
        12,
        'Who is Johannes Andreas August Grabau?',
        'Johannes Andreas August Grabau was a German-American Lutheran minister, theologian, and church leader. He was born on May 18, 1804, in Oldekop, Lower Saxony, Germany, and died on June 2, 1879, in Milwaukee, Wisconsin, United States. Grabau is best known for his conservative positions on church and ministry issues, which led him to break away from the Missouri Synod and found the Buffalo Synod and the Lutheran Church-Missouri Synod.',
        [True, MINOR, MAJOR],
        [
            'Johannes Andreas August Grabau was a German-American Lutheran minister.',
            'Grabau was also a theologian and a church leader.',
            'He was born on May 18, 1804, in Oldekop, Lower Saxony, Germany.',
            'Grabau died on June 2, 1879, in Milwaukee, Wisconsin, United States.',
            'He is best known for his conservative positions on church and ministry issues.',
            'These positions led him to break away from the Missouri Synod.',
            'He founded the Buffalo Synod and the Lutheran Church-Missouri Synod.'],
        [True, True, MINOR, MINOR, True, True, MAJOR]
    ],
    [
        13,
        'Who is Gündüz Kılıç?',
        'Gündüz Kılıç was a Turkish footballer, coach, and sports columnist. He was born on December 1, 1918 and died on July 4, 1980. Kılıç served as the coach of the Turkey national football team and numerous other clubs. He was known for his contributions to Turkish football and his emphasis on disciplined training.',
        [True, MINOR, True, MAJOR],
        [
            'Gündüz Kılıç was a Turkish footballer.',
            'He was also a coach.',
            'He was a sports columnist as well.',
            'He was born on December 1, 1918.',
            'He died on July 4, 1980.',
            'He served as the coach of the Turkey national football team.',
            'He also coached numerous other clubs.',
            'He was known for his contributions to Turkish football.',
            'He emphasized disciplined training.'
        ],
        [True, True, True, MINOR, MINOR, True, True, True, MAJOR]
    ],
    [
        14,
        'Who is Edward Synge?',
        'Edward Synge was an Anglican bishop in the Church of Ireland. He was born in 1659 and died in 1741. He served as the Bishop of Raphoe from 1714 to 1740 and the Bishop of Elphin from 1740 until his death in 1741. Synge is known for his writings on religious topics, with works such as "A Gentleman\'s Religion" and "Whole Duty of Man".',
        [True, MINOR, MAJOR, MAJOR],
        [
            'Edward Synge was an Anglican bishop in the Church of Ireland.',
            'He was born in 1659.',
            'He died in 1741.',
            'He served as the Bishop of Raphoe from 1714 to 1740.',
            'He served as the Bishop of Elphin from 1740 until his death in 1741.',
            'Synge is known for his writings on religious topics.',
            'He wrote works such as "A Gentleman\'s Religion" and "Whole Duty of Man".'
        ],
        [True, True, True, MINOR, MAJOR, True, MAJOR]
    ],
    [
        15,
        'Who is Whitey Kurowski?',
        'George John "Whitey" Kurowski was an American professional baseball player. He played as a third baseman in Major League Baseball from 1941 to 1949, all for the St. Louis Cardinals. Known for his batting prowess and defensive skills, Kurowski was a five-time All-Star and a World Series champion in 1942 and 1944. He was born on April 19, 1918 and died on October 17, 1999.',
        [True, True, MINOR, MINOR],
        [
            'George John "Whitey" Kurowski was an American professional baseball player.',
            'He played as a third baseman in Major League Baseball from 1941 to 1949.',
            'The entirety of his career was spent with the St. Louis Cardinals.',
            'Kurowski was known for his batting prowess and defensive skills.',
            'He was a five-time All-Star.',
            'He was a World Series champion in 1942 and 1944.',
            'He was born on April 19, 1918.',
            'He died on October 17, 1999.'
        ],
        [True, True, True, True, True, MINOR, True, MINOR]
    ],
    [
        16,
        'Who is John Holman (Chemist)?',
        'John Holman is a British chemist, education expert, and emeritus professor of chemistry at the University of Hull. He served as the president of the Royal Society of Chemistry from 2016 - 2019. He has made significant contributions to the fields of chemistry and chemistry education, having founded and directed the National Science Learning Centre. Dr. Holman is also the author of several chemistry textbooks. He was knighted in 2010 for his services to education.',
        [MINOR, MINOR, True, True, True],
        [
            'John Holman is a British chemist.',
            'He is an emeritus professor of chemistry at the University of Hull.',
            'He served as the president of the Royal Society of Chemistry from 2016 - 2019.',
            'He has made significant contributions to the fields of chemistry and chemistry education.',
            'He founded and directed the National Science Learning Centre.',
            'He is the author of several chemistry textbooks.',
            'He was knighted in 2010 for his services to education.',
        ],
        [True, MINOR, MINOR, True, True, True, True],
    ],
    [
        17,
        'Who is Guðlaugur Þór Þórðarson?',
        'Guðlaugur Þór Þórðarson is an Icelandic politician who has been the Minister for Foreign Affairs since November 2017. He is a member of the Independence Party. Prior to his current position, he served as Minister of Health from 2014 to 2017. He was first elected to the Althing (the national parliament of Iceland) in 2003.',
        [MINOR, True, MINOR, True],
        [
            'Guðlaugur Þór Þórðarson is an Icelandic politician.',
            'He has been the Minister for Foreign Affairs since November 2017.',
            'He is a member of the Independence Party.',
            'He served as Minister of Health from 2014 to 2017.',
            'He was first elected to the Althing (the national parliament of Iceland) in 2003.'
        ],
        [True, MINOR, True, MINOR, True],
    ],
    [
        18,
        'Who is Ha Jung-woo?',
        'Ha Jung-woo is a South Korean actor and film director. He is one of the leading actors in Korean cinema, known for his roles in films like "The Chaser," "The Yellow Sea," and "The Handmaiden." As a director, he has also made films under the pseudonym Yang Ik-june. Ha has won several awards for his acting, including Best Actor at the Blue Dragon Film Awards and the Grand Bell Awards.',
        [True, True, MAJOR, True],
        [
            'Ha Jung-woo is a South Korean actor and film director.',
            'He is one of the leading actors in Korean cinema.',
            'He is known for his roles in films like "The Chaser," "The Yellow Sea," and "The Handmaiden."',
            'As a director, he has made films under the pseudonym Yang Ik-june.',
            'He has won several awards for his acting.',
            'He has won the Best Actor at the Blue Dragon Film Awards.',
            'He has also won the Best Actor at the Grand Bell Awards.'
        ],
        [True, True, True, MAJOR, True, True, True],
    ],
    [
        19,
        'Who is Yaakov Israel Ifargan?',
        'Yaakov Israel Ifargan, also known as The Rentgen (or The X-Ray), is a famous Israeli Rabbi and Kabbalist who is known for his supposed divine abilities to see and diagnose medical issues. He is the brother of Rabbi Eliezer Berland and is considered one of the leading figures of the non-hasidic Haredi Judaism in Israel. He became famous in the 90s and is known for his spiritual gatherings and blessings, often participated in by politicians, businessmen, and celebrities.',
        [True, True, MAJOR, True],
        [
            'Yaakov Israel Ifargan is also known as The Rentgen or The X-Ray.',
            'He is a famous Israeli Rabbi and Kabbalist.',
            'Ifargan is known for his supposed divine abilities to see and diagnose medical issues.',
            'He is the brother of Rabbi Eliezer Berland.',
            'Ifargan is considered one of the leading figures of the non-hasidic Haredi Judaism in Israel.',
            'He became famous in the 90s.',
            'He is known for his spiritual gatherings and blessings.',
            'His gatherings and blessings are often participated in by politicians, businessmen, and celebrities.'
        ],
        [True, True, True, MAJOR, MAJOR, True, True, True]
    ],
    [
        20,
        'Who is Brian Petrovek?',
        'Brian Petrovek is a professional in the sports management industry. He was the Managing Owner and Chief Executive Officer for the Portland Pirates, a professional hockey team in the American Hockey League (AHL). Petrovek has also served as the CEO for USA Hockey’s Junior Program and the Adirondack Flames, among other roles. He has been recognized for his leadership in the industry, receiving awards such as the James C. Hendy Memorial Award for AHL’s outstanding executive. Petrovek has experience in both day-to-day management of teams and long-term strategic development.',
        [True, MAJOR, MAJOR, MAJOR, True],
        [
            'Brian Petrovek is a professional in the sports management industry.',
            'He was the Managing Owner and Chief Executive Officer for the Portland Pirates.',
            'The Portland Pirates is a professional hockey team in the American Hockey League (AHL).',
            'Petrovek has served as the CEO for USA Hockey’s Junior Program.',
            'He has also served as the CEO of the Adirondack Flames.',
            'He has been recognized for his leadership in the industry.',
            'He has received awards like the James C. Hendy Memorial Award for AHL’s outstanding executive.',
            "Petrovek's experience includes both day-to-day management of teams and long-term strategic development."
        ],
        [True, MAJOR, True, MAJOR, MAJOR, True, MAJOR, True]
    ],
]


assert all(len(datum[4]) == len(datum[5]) for datum in data)
assert all(False not in datum[3] for datum in data)
assert all(False not in datum[5] for datum in data)
