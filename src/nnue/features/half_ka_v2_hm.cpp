/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//Definition of input features HalfKAv2_hm of NNUE evaluation function

#include "half_ka_v2_hm.h"

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Features {

#if defined(USE_AVX512ICL)
void HalfKAv2_hm::write_indices(const std::array<Piece, SQUARE_NB>& oldPieces,
                                const std::array<Piece, SQUARE_NB>& newPieces,
                                Bitboard removedBB,
                                Bitboard addedBB,
                                Color perspective,
                                Square ksq,
                                ValueList<uint16_t, MaxActiveDimensions>& removed,
                                ValueList<uint16_t, MaxActiveDimensions>& added) {

    auto* write_removed = removed.make_space(popcount(removedBB));
    auto* write_added = added.make_space(popcount(addedBB));

    const __m512i vecOldPieces = _mm512_loadu_si512(oldPieces.data());
    const __m512i vecNewPieces = _mm512_loadu_si512(newPieces.data());

    const __m512i psi =
      _mm512_zextsi256_si512(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(PieceSquareIndex[perspective])));

    const __m512i allSquares = _mm512_set_epi8(
      63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
      47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
      31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
      15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1,  0
    );

    const uint16_t flip = 56 * perspective;
    const __m512i orient = _mm512_set1_epi16(OrientTBL[ksq] ^ flip);
    const __m512i bucket = _mm512_set1_epi16(KingBuckets[int(ksq) ^ flip]);

    auto make_indices = [&](Bitboard bb, __m512i board) {
        __m512i squares = _mm512_maskz_compress_epi8(bb, allSquares);
        __m512i pieces = _mm512_permutexvar_epi8(squares, board);
        squares = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(squares));
        pieces = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(pieces));
        const __m512i changed_psi = _mm512_permutexvar_epi16(pieces, psi);
        __m512i indices = _mm512_xor_si512(squares, orient);
        indices = _mm512_add_epi16(indices, changed_psi);
        indices = _mm512_add_epi16(indices, bucket);
        return indices;
    };

    const __m512i removed_indices = make_indices(removedBB, vecOldPieces);
    const __m512i added_indices = make_indices(addedBB, vecNewPieces);

    _mm512_storeu_si512(write_removed, removed_indices);
    _mm512_storeu_si512(write_added, added_indices);
}
#endif

// Index of a feature for a given king position and another piece on some square

IndexType HalfKAv2_hm::make_index(Color perspective, Square s, Piece pc, Square ksq) {
    const IndexType flip = 56 * perspective;
    return (IndexType(s) ^ OrientTBL[ksq] ^ flip) + PieceSquareIndex[perspective][pc]
         + KingBuckets[int(ksq) ^ flip];
}

// Get a list of indices for active features

void HalfKAv2_hm::append_active_indices(Color perspective, const Position& pos, IndexList& active) {
    Square   ksq = pos.square<KING>(perspective);
    Bitboard bb  = pos.pieces();
    while (bb)
    {
        Square s = pop_lsb(bb);
        active.push_back(make_index(perspective, s, pos.piece_on(s), ksq));
    }
}

// Get a list of indices for recently changed features

void HalfKAv2_hm::append_changed_indices(
  Color perspective, Square ksq, const DiffType& diff, IndexList& removed, IndexList& added) {
    removed.push_back(make_index(perspective, diff.from, diff.pc, ksq));
    if (diff.to != SQ_NONE)
        added.push_back(make_index(perspective, diff.to, diff.pc, ksq));

    if (diff.remove_sq != SQ_NONE)
        removed.push_back(make_index(perspective, diff.remove_sq, diff.remove_pc, ksq));

    if (diff.add_sq != SQ_NONE)
        added.push_back(make_index(perspective, diff.add_sq, diff.add_pc, ksq));
}

bool HalfKAv2_hm::requires_refresh(const DiffType& diff, Color perspective) {
    return diff.pc == make_piece(perspective, KING);
}

}  // namespace Stockfish::Eval::NNUE::Features
