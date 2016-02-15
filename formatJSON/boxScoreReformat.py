#from pymongo import MongoClient
import pprint
import pymongo


client = pymongo.MongoClient("mongodb://localhost:27017")
db = client.nba
#db = client.test

def agg1():
    result = db.sigOptBox.aggregate([
    {"$match":{"resultSets.headers":{"$in":["GAME_ID",
						"TEAM_ID",
						"TEAM_NAME",
						"TEAM_ABBREVIATION",
						"TEAM_CITY",
						"MIN",
						"FGM",
						"FGA",
						"FG_PCT",
						"FG3M",
						"FG3A",
						"FG3_PCT",
						"FTM",
						"FTA",
						"FT_PCT",
						"OREB",
						"DREB",
						"REB",
						"AST",
						"STL"
							]
					  
			}
		}
    }, 
    {"$project":{
		"resultSets":{"$arrayElemAt":["$resultSets", 5]}
		}
    }	,						
    {"$unwind":"$resultSets.rowSet"},
    {"$project":{ 
		"_id":0,
		"GAME_ID":{"$arrayElemAt":["$resultSets.rowSet",0]}, 
		"TEAM_ID":{"$arrayElemAt":["$resultSets.rowSet",1]}, 
		"TEAM_NAME":{"$arrayElemAt":["$resultSets.rowSet",2]}, 
		"TEAM_ABBREVIATION":{"$arrayElemAt":["$resultSets.rowSet",3]}, 
		"TEAM_CITY":{"$arrayElemAt":["$resultSets.rowSet",4]}, 
		"MIN":{"$arrayElemAt":["$resultSets.rowSet",5]}, 
		"FGM":{"$arrayElemAt":["$resultSets.rowSet",6]}, 
		"FGA":{"$arrayElemAt":["$resultSets.rowSet",7]}, 
		"FG_PCT":{"$arrayElemAt":["$resultSets.rowSet",8]}, 
		"FG3M":{"$arrayElemAt":["$resultSets.rowSet",9]}, 
		"FG3A":{"$arrayElemAt":["$resultSets.rowSet",10]}, 
		"FG3_PCT":{"$arrayElemAt":["$resultSets.rowSet",11]}, 
		"FTM":{"$arrayElemAt":["$resultSets.rowSet",12]}, 
		"FTA":{"$arrayElemAt":["$resultSets.rowSet",13]}, 
		"FT_PCT":{"$arrayElemAt":["$resultSets.rowSet",14]}, 
		"OREB":{"$arrayElemAt":["$resultSets.rowSet",15]}, 
		"DREB":{"$arrayElemAt":["$resultSets.rowSet",16]}, 
		"REB":{"$arrayElemAt":["$resultSets.rowSet",17]}, 
		"AST":{"$arrayElemAt":["$resultSets.rowSet",18]}, 
		"STL":{"$arrayElemAt":["$resultSets.rowSet",19]}, 
		"BLK":{"$arrayElemAt":["$resultSets.rowSet",20]}, 
		"TO":{"$arrayElemAt":["$resultSets.rowSet",21]}, 
		"PF":{"$arrayElemAt":["$resultSets.rowSet",22]}, 
		"PTS":{"$arrayElemAt":["$resultSets.rowSet",23]}, 
		"PLUS_MINUS":{"$arrayElemAt":["$resultSets.rowSet",24]}, 
		 } 
    },
    {"$out":"test"}
   ])

    return result

if __name__ == '__main__':
	result1 = agg1()
	for document in result1:
		print(document)
                exit()


