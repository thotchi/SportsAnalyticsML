CREATE TABLE box_master
as
select
cast(game_id as INT)
, cast(substring(date from 1 for 10) as date) date_game
, cast(season as INT) season
, cast(team_abbreviation as VARCHAR) team_abbreviation
, cast(team_city as VARCHAR) team_city
, cast(team_id as INT) team_id
, cast(team_name as VARCHAR) team_name
, cast(home_away as VARCHAR) home_away
, cast(turnovers as INT) turnovers
, cast(cast(substring(min from 1 for position(':' in min)-1) as NUMERIC(5,2)) + cast(substring(min from position(':' in min)+1 for 2) as NUMERIC(5,2))/60 as numeric(5,2)) total_minutes
--, cast(_id as VARCHAR)
, rank() OVER (PARTITION BY team_id ORDER BY cast(substring(date from 1 for 10) as date) DESC) team_game_rank 
, cast(ast as INT) ast
, cast(blk as INT) blk
, cast(dreb as INT) dreb
, cast(oreb as INT) oreb
, cast(reb as INT) reb
, cast(stl as INT) stl
, cast(pts as NUMERIC(4,1)) pts
, cast(ptsa as NUMERIC(4,1)) ptsa
, cast(plus_minus as NUMERIC(4,1)) pt_diff
, cast(fg3a as INT) fg3a
, cast(fg3m as INT) fg3m
, cast(fg3_pct as NUMERIC(4,3)) 
, cast(fga as INT) fga
, cast(fgm as INT) fgm
, cast(fg_pct as NUMERIC(4,3))
, cast(fta as INT) fta
, cast(ftm as INT) ftm
, cast(ft_pct as NUMERIC(4,3))
, cast(pf as INT) pf
from boxscore
order by team_id, date_game desc