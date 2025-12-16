from neo4j import GraphDatabase


class BaselineRetriever:

    def __init__(self, driver):
        self.driver = driver

    
    def flights_from_airport(self, origin_code):
        query = """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(a:Airport {station_code:$code})
        RETURN f.flight_number AS flight, a.station_code AS origin
        """
        with self.driver.session() as session:
            return session.run(query, code=origin_code).data()


    def flights_to_airport(self, dest_code):
        query = """
        MATCH (f:Flight)-[:ARRIVES_AT]->(a:Airport {station_code:$code})
        RETURN f.flight_number AS flight, a.station_code AS destination
        """
        with self.driver.session() as session:
            return session.run(query, code=dest_code).data()


    def passenger_journeys(self, record_locator):
        query = """
        MATCH (p:Passenger {record_locator:$rl})-[:TOOK]->(j:Journey)
        RETURN j.feedback_ID AS journey_id, j.arrival_delay_minutes AS delay
        """
        with self.driver.session() as session:
            return session.run(query, rl=record_locator).data()


    def journey_flight(self, feedback_id):
        query = """
        MATCH (j:Journey {feedback_ID:$fid})-[:ON]->(f:Flight)
        RETURN j.feedback_ID AS journey_id, f.flight_number AS flight
        """
        with self.driver.session() as session:
            return session.run(query, fid=feedback_id).data()

    
    def flights_between(self, origin, dest):
        query = """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(o:Airport {station_code:$orig})
        MATCH (f)-[:ARRIVES_AT]->(d:Airport {station_code:$dest})
        RETURN f.flight_number AS flight
        """
        with self.driver.session() as session:
            return session.run(query, orig=origin, dest=dest).data()

   
    def passengers_on_flight(self, flight_number):
        query = """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)-[:ON]->(f:Flight {flight_number:$fnum})
        RETURN p.record_locator AS passenger
        """
        with self.driver.session() as session:
            return session.run(query, fnum=flight_number).data()

    
    def flights_by_fleet(self, fleet_type):
        query = """
        MATCH (f:Flight {fleet_type_description:$fleet})
        RETURN f.flight_number AS flight
        """
        with self.driver.session() as session:
            return session.run(query, fleet=fleet_type).data()


    def food_scores_by_passenger(self, record_locator):
        query = """
        MATCH (p:Passenger {record_locator:$record_locator})-[:TOOK]->(j:Journey)
        RETURN j.feedback_ID AS journey_id,
               j.food_satisfaction_score AS food_score
        ORDER BY food_score DESC
        """
        with self.driver.session() as session:
            return session.run(query, record_locator=record_locator).data()

 
    def top_food_flights(self, k):
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WITH f.flight_number AS flight,
             avg(j.food_satisfaction_score) AS avg_food
        RETURN flight, avg_food
        ORDER BY avg_food DESC
        LIMIT $k
        """
        with self.driver.session() as session:
            return session.run(query, k=k).data()


    def passengers_by_generation(self, gen):
        query = """
        MATCH (p:Passenger {generation:$g})
        RETURN p.record_locator AS passenger
        """
        with self.driver.session() as session:
            return session.run(query, g=gen).data()

   
    def long_flights(self, min_miles):
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)
        WHERE j.actual_flown_miles > $m
        RETURN DISTINCT f.flight_number AS flight
        """
        with self.driver.session() as session:
            return session.run(query, m=min_miles).data()

    
    def airports_used_by_passenger(self, record_locator):
        query = """
        MATCH (p:Passenger {record_locator:$rl})-[:TOOK]->(j:Journey)-[:ON]->(f:Flight)
        MATCH (f)-[:DEPARTS_FROM]->(a1:Airport)
        MATCH (f)-[:ARRIVES_AT]->(a2:Airport)
        RETURN DISTINCT a1.station_code AS origin, a2.station_code AS destination
        """
        with self.driver.session() as session:
            return session.run(query, rl=record_locator).data()
