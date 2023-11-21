sql_query = '''
    WITH silver_rooms AS (
    SELECT
            room_id,
            type,
            buyin,
            room_seasonality,
            max_user_lineups,
            category,
            room_capacity,
            championship_group
        FROM `everest-bigquery.dbt_dfs_silver.rooms`
        WHERE room_start_datetime >= current_date - 30
    ),
    user_entrances AS (
        SELECT
        user_id,
        room_id,
        lineup_id,
        lineup_create_date
        FROM `everest-bigquery.dbt_dfs_silver.lineups`
        WHERE lineup_create_date >= current_date - 30
    )

    SELECT
    ue.user_id,
    sr.type,
    sr.buyin,
    sr.room_seasonality,
    sr.max_user_lineups,
    sr.category,
    sr.room_capacity,
    sr.championship_group,
    FROM user_entrances ue
    INNER JOIN silver_rooms sr
    ON ue.room_id = sr.room_id
'''

active_rooms_sql_query = '''
    SELECT
        room_id,
        type,
        buyin,
        room_seasonality,
        max_user_lineups,
        category,
        room_capacity,
        championship_group
    FROM `everest-bigquery.dbt_dfs_silver.rooms`
    WHERE active is TRUE
    AND room_last_entrance_datetime >  timestamp_add(current_datetime, INTERVAL 5 minute)

'''