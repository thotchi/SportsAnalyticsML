import grequests
import os
import json
import math
import traceback

"""
This script scrapes a subset of games of a given NBA season based on two 
arguments: First argument: The subset of games to scrape defined by 
NUM_GAMES_A_SEASON (1230) and the "total". Currently, it can range from 1 to 7
Second argument: The first year of the NBA season without millenium or century 
(e.g. the 2013-14 season would be "13")
"""

NUM_GAMES_A_SEASON = 82 * 30 / 2
SEASONS_RANGE = range(0, 16)
GAMES_RANGE = range(1, NUM_GAMES_A_SEASON + 1)

def scrape(start, stop, season):
    DATA_DIR = 'data'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    def passer(request, e, **kwargs):
        pass

    def save_json(response, **kwargs):
        print response; # I GET <Response [400]> WHICH SUGGESTS THAT THE
                        # THE SERVER RETURNED AN ERROR
                        #
                        # 10.4.1 400 Bad Request
                        # The request could not be understood by the server 
                        # due to malformed syntax. The client SHOULD NOT 
                        # repeat the request without modifications.

        boxscore_json = response.json()

        # DOESN'T MAKE IT PAST response.json().  SOMEHOW, THIS RETURNS AFTER
        # CALLING THIS.
        
#        try:
        if boxscore_json.get('Message') == 'An error has occurred.':
            return
        result_sets = boxscore_json['resultSets']
        date = result_sets[0]["rowSet"][0][0].split("T")[0]
        game_id = boxscore_json['parameters']['GameID']
        unique_id = "{}-{}.json".format(date, game_id)
        with open(os.path.join(DATA_DIR, unique_id), 'w') as outfile:
            json.dump(boxscore_json, outfile)
#        except Exception as e:
#            traceback.print_exc()
#        finally:
#            response.close()

    async_list = []
    for game_num in GAMES_RANGE[start:stop]:
        game_id = "002{:02}0{:04}".format(season, game_num)
        url = 'http://stats.nba.com/stats/boxscore/?GameId={}'.format(game_id);
        url += '&StartPeriod=0&EndPeriod=0&StartRange=0&EndRange=0';
        url += '&RangeType=0';

        print url;  # URL LOOKS LIKE IT'S OKAY.  I CAN USE THE URL IN THE 
                    # BROWSER AND CAN DOWNLOAD A JSON FILE.

        action_item = grequests.get(url, hooks={'response': save_json})
        async_list.append(action_item)

    grequests.map(
        async_list
        ,size=len(async_list)
        ,stream=False
        ,exception_handler=passer
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'start'
        ,type=int
        ,help='The index in the season to start scraping.'
    )
    parser.add_argument(
        'stop'
        ,type=int
        ,help='The index in the season to stop scraping.'
    )
    parser.add_argument(
        'season'
        ,type=int
        ,help='The season to scrape (for example: 13 for the 2013-2014 season)'
        ,nargs='?'
    )
    args = parser.parse_args()

    try:
        if not (0 <= int(args.season) <= 16):
            raise ValueError()
    except (TypeError, ValueError):
        raise Exception('Season must be a number between 0 and 15 ' + \
            '(for example: 13 for the 2013-2014 season)'
            )

    scrape(args.start, args.stop, args.season)
