-- DROP DATABASE IF EXISTS GenderIdentification;
SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';


-- Schema GenderIdentification
CREATE SCHEMA IF NOT EXISTS `GenderIdentification`;
USE `GenderIdentification`;


-- Table GenderIdentification.facial_landmarks
CREATE TABLE IF NOT EXISTS `GenderIdentification`.`facial_landmarks` (
    id SERIAL PRIMARY KEY,
    p_id varchar(10) NOT NULL,   -- m-001
    image_id varchar(10) NOT NULL,  -- m-001-01.pts
    x_coordinate FLOAT NOT NULL, 
    y_coordinate FLOAT NOT NULL,
    PRIMARY KEY (`id`)
);




