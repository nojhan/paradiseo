/*
(c) Thales group, 2012

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation;
    version 2 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
Contact: http://eodev.sourceforge.net

Authors:
    Benjamin Bouvier <benjamin.bouvier@gmail.com>
*/
# ifndef __EOSERIAL_PARSER_H__
# define __EOSERIAL_PARSER_H__

# include "Entity.h"
# include "SerialString.h"
# include "SerialObject.h"

/**
 * @file Parser.h
 *
 * This file contains a tiny JSON parser used in DAE. This parser just handles 
 * a subset of JSON grammar, with the following restrictions :
 * - all strings must be surrounded by double quotes.
 * - everything which is not an object or an array is considered to be a string
 * (even integers, booleans,...).
 * - no syntax check is done. We trust the programmer and he has to ensure that
 *   every JSON string he produces is valid.
 *
 * @author Benjamin BOUVIER
 */

namespace eoserial
{

/**
 * @brief Parser from a JSON source.
 *
 * This parser does just retrieve values and does NOT check the structure of
 * the input. This implies that if the input is not correct, the result is undefined
 * and can result to a failure on execution.
 *
 * @ingroup Serialization
 */
class Parser
{
    public:

        /**
         * @brief Parses the given string and returns the JSON object read.
         */
        static eoserial::Object* parse(const std::string & str);

    protected:

        /**
         * @brief Parses the right part of a JSON object as a string.
         *
         * The right part of an object can be a string (for instance :
         * "key":"value"), a JSON array (for instance: "key":["1"]) or
         * another JSON object (for instance: "key":{"another_key":"value"}).
         *
         * The right parts are found after keys (which are parsed by parseLeft)
         * and in arrays.
         *
         * @param str The string we're parsing.
         * @param pos The index of the current position in the string.
         * @return The JSON object matching the right part.
         */
        static eoserial::Entity* parseRight(const std::string & str, size_t & pos);

        /**
         * @brief Parses the left value of a key-value pair, which is the key.
         *
         * @param str The string we're parsing.
         * @param pos The index of the current position in the string.
         * @param json The current JSON object for which we're adding a key-value pair.
         */
        static void parseLeft(const std::string & str, size_t & pos, eoserial::Object* json);

        /**
         * @brief Retrieves a string in a JSON content.
         *
         * @param str The string we're parsing.
         * @param pos The index of the current position of parsing,
         * which will be updated.
         */
        static eoserial::String* parseJsonString(const std::string & str, size_t & pos);
};

} // namespace eoserial

# endif // __EOSERIAL_PARSER_H__
