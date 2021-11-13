#![warn(missing_docs)]
// Copyright 2021 Daniel May
// 
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! A small and easy to use library to generate mazes for games
//!
//! Irrgarten enables you to procedurally generate mazes of arbitrary size. Compared to similar
//! libraries, Irrgarten puts special focus on using these mazes within games. The main
//! advantages are:
//!
//! ### The generated mazes are essentially tilemaps - and walls are just tiles
//!
//! This is different to other generators that create beautiful mazes but use internal representations
//! (such as bitmasks) for walls. With Irrgarten you just get a two dimensional vector
//! like this:
//!
//! ```
//! 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
//! 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 
//! 1 0 1 1 1 0 1 1 1 1 1 1 1 0 1 
//! 1 0 1 0 0 0 1 0 0 0 1 0 0 0 1 
//! 1 0 1 0 1 1 1 0 1 0 1 0 1 1 1 
//! 1 0 0 0 1 0 0 0 1 0 1 0 0 0 1 
//! 1 1 1 1 1 0 1 1 1 0 1 1 1 0 1 
//! 1 0 0 0 1 0 0 0 1 0 1 0 0 0 1 
//! 1 0 1 0 1 1 1 0 1 1 1 0 1 1 1 
//! 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 
//! 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 
//! 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 
//! 1 0 1 1 1 1 1 0 1 1 1 1 1 0 1 
//! 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 
//! 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
//! ```
//!
//! With walls being ```1``` and the floor being ```0``` you can just 1:1 map this to
//! any tilemap, gridmap or whatever tile-like graphical structure your favourite engine
//! offers.
//!
//! ### Injectable randomness
//!
//! You provide your own random generator. Thus, you are in full control
//! of the seed and the internal state of the randomness. This enables you to
//! deterministically generate the same mazes, for example, across the network. Instead
//! of synchronizing the whole maze, you now only have to synchronize the seed between
//! the peers. You can also use this seed for replay systems, let testers share difficult
//! seeds with you to evaluate etc. Invaluable for games with this kind of procedural
//! content.
//!
//! # Usage
//!
//! Add this to your Cargo.toml
//!
//! ```toml
//! [dependencies]
//! irrgarten = "0.1"
//! ```
//!
//! You will also need to choose a random number generator. For the following example,
//! we simply use the [rand](https://crates.io/crates/rand) crate.
//!
//! # Example: Simple maze generation
//!
//! ```rust
//! use irrgarten::Maze;
//! use rand;
//!
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!     
//!     // Full maze generation. The dimensions can be any odd number that is > 5.
//!     let maze = Maze::new(63, 31).unwrap().generate(&mut rng);
//!     
//!     // The generated maze data can be accessed via index:
//!     for y in 0..maze.height {
//!         for x in 0..maze.width {
//!             println!("{}", maze[x][y]);
//!         }
//!     }
//!     // Alternatively, Maze also provides into_iter(), iter() and iter_mut()
//!     // methods to iterate over the columns.
//! }
//! ```
//!
//! # Example: Generate an image of the maze (for easy inspection/prototyping)
//!
//! ```rust
//! use irrgarten::Maze;
//! use rand;
//! use std::fs;
//!
//! fn main() {
//!     let mut rng = rand::thread_rng();
//!     let maze = Maze::new(255, 255).unwrap().generate(&mut rng);
//!
//!     // Save image to disk. Can be opened with most image viewers.
//!     fs::write("maze.pbm", maze.to_pbm()).unwrap();
//! }
//! ```
//!
//! # Example: Use a different random number generator with a seed
//!
//! Using the [xoshiro](https://crates.io/crates/rand_xoshiro) generator:
//!
//! ```rust
//! use irrgarten::Maze;
//! use rand_xoshiro::rand_core::SeedableRng;
//! use rand_xoshiro::Xoshiro256Plus;
//!
//! fn main() {
//!     let mut rng = Xoshiro256Plus::seed_from_u64(123123123);
//!     let maze = Maze::new(4095, 4095).unwrap().generate(&mut rng);
//! }
//! ```

use rand::{prelude::SliceRandom, Rng};
use std::convert::TryInto;
use std::ops::{Add, Index, IndexMut, Mul};
use std::slice::{Iter, IterMut};
use thiserror::Error;

const TILE_FLOOR: u8 = 0;
const TILE_WALL: u8 = 1;

/// Error enum for maze generation
#[derive(Error, Debug, PartialEq, Eq)]
pub enum MazeGenerationError {
    /// Maze dimensions must be odd and >= 5
    #[error("maze dimensions must be odd and >= 5")]
    InvalidDimensions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Direction {
    North,
    East,
    South,
    West,
}

const ALL_DIRS: [Direction; 4] = [
    Direction::North,
    Direction::East,
    Direction::South,
    Direction::West,
];

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct TinyVec {
    x: isize,
    y: isize,
}

impl Add for TinyVec {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Mul<isize> for TinyVec {
    type Output = Self;

    fn mul(self, rhs: isize) -> Self::Output {
        Self::Output {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

/// Maze structure that contains all maze data
///
/// The maze's dimensions need to be odd numbers >= 5. After creation,
/// call [generate](#method.generate) to generate the data.
/// You can access the generated maze data by indexing (e.g. ```maze[x][y]```) or by
/// using [iter](#method.iter), [into_iter](#method.into_iter)
/// or [iter_mut](#method.iter_mut).
#[derive(Debug, Clone, PartialEq)]
pub struct Maze {
    /// Width of the maze. Must be an odd number >= 5.
    pub width: usize,
    /// Height of the maze. Must be an odd number >= 5.
    pub height: usize,

    data: Vec<Vec<u8>>
}

impl Index<usize> for Maze {
    type Output = Vec<u8>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Maze {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl IntoIterator for Maze {
    type Item = Vec<u8>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl Maze {
    /// Construct the maze. Only odd values >= 5 can be passed.
    pub fn new(width: usize, height: usize) -> Result<Self, MazeGenerationError> {
        if width < 5 || width % 2 == 0 || height < 5 || height % 2 == 0 {
            return Err(MazeGenerationError::InvalidDimensions);
        }
        Ok(Self {
            width,
            height,
            data: vec![vec![TILE_WALL; height]; width]
        })
    }

    /// Iterate over the maze data column-wise.
    pub fn iter(&self) -> Iter<Vec<u8>> {
        self.data.iter()
    }

    /// Mutably iterate over the maze data column-wise.
    pub fn iter_mut(&mut self) -> IterMut<Vec<u8>> {
        self.data.iter_mut()
    }

    /// Generate the maze data.
    pub fn generate<R>(mut self, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        // This basically uses a recursive backtracking algorithm. However, it was
        // modified to be iterative. Since we model walls as tiles, we need
        // to carve two cells per iteration (carve through the walls).
        let mut stack = Vec::new();

        let start = TinyVec { x: 1, y: 1 };
        stack.push((start, start));

        while !stack.is_empty() {
            let (path, cell) = stack.pop().unwrap();

            if self.data[cell.x as usize][cell.y as usize] == TILE_WALL {
                // clear the path up to the cell
                self.data[path.x as usize][path.y as usize] = TILE_FLOOR;
                self.data[cell.x as usize][cell.y as usize] = TILE_FLOOR;

                // go in random direction
                let mut dirs = ALL_DIRS.clone();
                dirs.shuffle(rng);
                for dir in dirs {
                    let step = match dir {
                        Direction::North => TinyVec { x: 0, y: -1 },
                        Direction::East => TinyVec { x: 1, y: 0 },
                        Direction::South => TinyVec { x: 0, y: 1 },
                        Direction::West => TinyVec { x: -1, y: 0 },
                    };
                    let double = cell + step * 2;

                    // check, if this path is valid
                    if double.x >= 0
                        && double.x < self.width.try_into().unwrap()
                        && double.y >= 0
                        && double.y < self.height.try_into().unwrap()
                        && self.data[double.x as usize][double.y as usize] == TILE_WALL
                    {
                        // continue carving from there
                        stack.push((cell + step, double));
                    }
                }
            }
        }
        self
    }

    /// Generate pbm image data. Can be saved to disk.
    pub fn to_pbm(&self) -> String {
        let mut pbm = String::from("P1\n");
        pbm.push_str(&format!("{} {}\n", self.width, self.height));
        for y in 0..self.height {
            for x in 0..self.width {
                pbm.push_str(&format!("{} ", self.data[x][y]));
            }
            pbm.push_str("\n");
        }
        pbm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensions() {
        assert_eq!(Maze::new(3, 7), Err(MazeGenerationError::InvalidDimensions));
        assert_eq!(Maze::new(9, 6), Err(MazeGenerationError::InvalidDimensions));

        let (w, h) = (7, 5);
        let maze = Maze::new(w, h);
        assert_eq!(maze.is_ok(), true);
        if let Ok(maze) = maze {
            assert_eq!(maze.width, w);
            assert_eq!(maze.data.len(), w);
            assert_eq!(maze.height, h);
            assert_eq!(maze.data[0].len(), h);
        }
    }

    #[test]
    fn test_generation() {
        use rand::thread_rng;

        let mut rng = thread_rng();
        let maze = Maze::new(13, 13).unwrap().generate(&mut rng);

        // make sure that something was actually generated
        assert_eq!(maze.iter().flatten().all(|&tile| tile == TILE_WALL), false);

        // make sure that shape didn't change
        assert_eq!(maze.data.len(), maze.width);
        assert_eq!(maze.data[0].len(), maze.height);

        // make sure that maze is surrounded by walls
        for y in 0..maze.height {
            for x in 0..maze.width {
                if y == 0 || y == maze.height - 1 {
                    assert_eq!(maze[x][y], TILE_WALL);
                } else if x == 0 || x == maze.width - 1 {
                    assert_eq!(maze[x][y], TILE_WALL);
                }
            }
        }
    }

    #[test]
    fn test_pbm() {
        use rand::thread_rng;

        let mut rng = thread_rng();
        let (w, h) = (11, 13);
        let maze = Maze::new(w, h).unwrap().generate(&mut rng);
        let pbm = maze.to_pbm();

        // check header
        assert_eq!(pbm.starts_with(&format!("P1\n{} {}\n", w, h)), true);

        // check data
        let data: Vec<&str> = pbm.lines().skip(2).collect();
        for y in 0..maze.height {
            let tiles: Vec<u8> = data[y]
                .split_whitespace()
                .map(|t| t.parse::<u8>().unwrap())
                .collect();
            for x in 0..maze.width {
                assert_eq!(tiles[x], maze[x][y]);
            }
        }
    }
}
