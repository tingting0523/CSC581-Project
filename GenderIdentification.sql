-- DROP DATABASE IF EXISTS GenderIdentification;
SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';


-- Schema GenderIdentification
CREATE SCHEMA IF NOT EXISTS `GenderIdentification`;
USE `GenderIdentification`;


-- Table GenderIdentification.facial_landmarks
drop table if exists `GenderIdentification`.`facial_landmarks`;
CREATE TABLE IF NOT EXISTS `GenderIdentification`.`facial_landmarks` (
    id SERIAL PRIMARY KEY,
    p_id varchar(10) NOT NULL,   -- m-001
    gender int Not null,  -- 0:male  1:female
    image_id varchar(10) NOT NULL,  -- m-001-01.pts
    point_id INT NOT NULL,  -- 0/1/2....
    x_coordinate FLOAT NOT NULL, 
    y_coordinate FLOAT NOT NULL 
);
ALTER TABLE GenderIdentification.facial_landmarks ADD CONSTRAINT unique_landmark UNIQUE (p_id, image_id, point_id);




